import os
import sys
from typing import List, Dict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------- logging
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_dir = "convtrans_logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
log_file = open(log_path, "w", buffering=1)
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

# ============================================================
# helper functions
# ============================================================
def numpy_normalised_quantile_loss(y: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    diff = y - y_pred
    werr = quantile * np.maximum(diff, 0.0) + (1.0 - quantile) * np.maximum(-diff, 0.0)
    ql = werr.mean()
    normaliser = np.abs(y).mean()
    return 2 * ql / (normaliser + 1e-8)

def smape_np(y_true, y_pred):
    return np.mean(200.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

# ============================================================
# formatter & dataset
# ============================================================
class CitibikeFormatterLite:
    def __init__(self):
        self.identifier_cols = ["station_id", "time"]
        self.target_col = "values"

        self.observed_input_cols = [
            "arrivals", "departures",
            "spd_med_lag6", "tt_med", "wspd_obs_roll6h_mean",
            "pm25_mean_lag24", "pm25_mean_roll3h_mean", "spd_med_roll24h_sum",
            "pm25_mean", "pm25_mean_lag3", "temp_obs_roll24h_mean",
            "wspd_obs_roll6h_sum", "pm25_mean_lag6",
            "pm25_mean_roll24h_mean", "pm25_mean_roll6h_sum", "pm25_mean_roll3h_sum",
            "temp_obs_roll24h_sum", "rh_obs", "pm25_mean_lag1",
            "pm25_mean_roll24h_sum", "wspd_obs_roll24h_mean",
        ]

        self.known_input_cols = ["hour", "dow", "is_weekend", "sensor_day", "is_night"]

        self.static_input_cols = ["latitude", "longitude", "station_name", "station_id"]

        self.categorical_cols = ["station_id", "dow", "is_weekend", "is_night", "station_name"]

        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def _ensure_core_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        expected = (
            self.identifier_cols
            + [self.target_col]
            + self.observed_input_cols
            + self.known_input_cols
            + self.static_input_cols
        )
        for c in expected:
            if c not in df.columns:
                df[c] = np.nan
        return df

    def _propose_boundaries(self, num_days: int):
        if num_days <= 10:
            valid_b = max(3, num_days - 4)
            test_b  = max(5, num_days - 2)
        else:
            valid_b = int(round(num_days * 0.7))
            test_b  = int(round(num_days * 0.85))
        valid_b = max(3, min(valid_b, num_days - 4))
        test_b  = max(valid_b + 1, min(num_days - 1, num_days))
        return int(valid_b), int(test_b)

    def split_data(self, df, valid_boundary=None, test_boundary=None):
        print('Formatting train-valid-test splits for full Citi Bike dataset.')
        df = self._ensure_core_columns(df)
        index = df['sensor_day']
        num_days = int(index.max() - index.min() + 1)

        valid_boundary = 22
        test_boundary = 26

        if valid_boundary in (None, 'auto') or test_boundary in (None, 'auto'):
            valid_boundary, test_boundary = self._propose_boundaries(num_days)
            print(f"[CitibikeFormatterLite] Auto boundaries → valid={valid_boundary}, test={test_boundary}, num_days={num_days}")

        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
        test  = df.loc[index >= test_boundary - 7]

        self.set_scalers(train)
        train_t, valid_t, test_t = (self.transform_inputs(x) for x in [train, valid, test])

        print(f"Train set size: {len(train_t)} rows")
        print(f"Valid set size: {len(valid_t)} rows")
        print(f"Test  set size: {len(test_t)} rows")
        return train_t, valid_t, test_t

    def set_scalers(self, df: pd.DataFrame):
        continuous_cols = (
            [self.target_col]
            + self.observed_input_cols
            + [c for c in self.known_input_cols if c not in self.categorical_cols]
            + [c for c in self.static_input_cols if c not in self.categorical_cols]
        )
        for c in continuous_cols:
            scaler = StandardScaler()
            vals = df[c].astype(float).values.reshape(-1, 1)
            scaler.fit(vals)
            self.scalers[c] = scaler

        for c in self.categorical_cols:
            le = LabelEncoder()
            vals = df[c].astype(str).fillna("__MISSING__").values
            le.fit(vals)
            self.label_encoders[c] = le

    def transform_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c, scaler in self.scalers.items():
            v = df[c].astype(float).fillna(df[c].astype(float).mean())
            df[c] = scaler.transform(v.values.reshape(-1,1))
        for c, le in self.label_encoders.items():
            v = df[c].astype(str).fillna("__MISSING__")
            v = v.map(lambda x: x if x in le.classes_ else "__MISSING__")
            if "__MISSING__" not in le.classes_:
                le.classes_ = np.append(le.classes_, "__MISSING__")
            df[c] = le.transform(v)
        return df

class CitibikeSeqDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        formatter: CitibikeFormatterLite,
        encoder_length: int = 168,
        forecast_horizon: int = 24,
        split_name: str = "train",
    ):
        self.df = df.sort_values(["station_id","time"]).reset_index(drop=True)
        self.fmt = formatter
        self.encoder_length = encoder_length
        self.forecast_horizon = forecast_horizon
        self.split_name = split_name

        self.feature_cols = (
            formatter.observed_input_cols
            + formatter.known_input_cols
            + formatter.static_input_cols
        )
        self.target_col = formatter.target_col

        self.indices = []
        self._build_indices()

    def _build_indices(self):
        enc = self.encoder_length
        dec = self.forecast_horizon
        total = enc + dec

        for _, sub in self.df.groupby("station_id"):
            sub = sub.sort_values("time")
            sub_idx = sub.index.to_list()
            values_len = len(sub_idx)
            if values_len < total:
                continue
            for local_start in range(0, values_len - total + 1):
                dec_rows_global = sub_idx[local_start + enc : local_start + total]
                window_target = self.df.loc[dec_rows_global, self.target_col]
                if window_target.isna().all():
                    continue
                self.indices.append(sub_idx[local_start])

        print(f"[{self.split_name}] built total windows: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        enc_end = start + self.encoder_length
        dec_end = enc_end + self.forecast_horizon

        enc_df = self.df.iloc[start:enc_end]
        dec_df = self.df.iloc[enc_end:dec_end]

        static_vals = enc_df[self.fmt.static_input_cols].iloc[0].values.astype(np.float32)
        static_enc = np.repeat(static_vals[None,:], self.encoder_length, axis=0)
        static_dec = np.repeat(static_vals[None,:], self.forecast_horizon, axis=0)

        enc_feat = enc_df[self.fmt.observed_input_cols + self.fmt.known_input_cols].values.astype(np.float32)
        enc_feat = np.concatenate([enc_feat, static_enc], axis=1)

        dec_known = dec_df[self.fmt.known_input_cols].values.astype(np.float32)
        dec_feat = np.concatenate([dec_known, static_dec], axis=1)

        target = dec_df[self.target_col].values.astype(np.float32)

        return (
            torch.from_numpy(enc_feat),
            torch.from_numpy(dec_feat),
            torch.from_numpy(target),
        )

# ============================================================
# ConvTrans
# ============================================================
class ConvSelfAttention(nn.Module):
    def __init__(self, dim_model, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_q = nn.Conv1d(dim_model, dim_model, kernel_size, padding=kernel_size-1, bias=False)
        self.conv_k = nn.Conv1d(dim_model, dim_model, kernel_size, padding=kernel_size-1, bias=False)
        # note: padding ensures causal conv if we trim later or mask accordingly
        self.scale = dim_model ** -0.5

    def forward(self, x):
        """
        x: [B, T, D] → transpose → [B, D, T] for conv1d
        returns attention scores: [B, T, T]
        """
        B, T, D = x.size()
        x_t = x.transpose(1, 2)  # [B, D, T]
        q = self.conv_q(x_t)[:, :, :T]  # trim excess padding → [B, D, T]
        k = self.conv_k(x_t)[:, :, :T]  # [B, D, T]
        q = q.transpose(1, 2)  # [B, T, D]
        k = k.transpose(1, 2)  # [B, T, D]
        scores = torch.matmul(q, k.transpose(1,2)) * self.scale  # [B, T, T]
        return scores

class TransformerLayerConv(nn.Module):
    def __init__(self, dim_model, nhead, kernel_size=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.conv_sa = ConvSelfAttention(dim_model, kernel_size=kernel_size)
        self.nhead = nhead
        self.dim_model = dim_model
        self.attn_drop = nn.Dropout(dropout)
        self.proj_v = nn.Linear(dim_model, dim_model)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.ff = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # x: [B, T, D]
        # compute conv self-attention
        scores = self.conv_sa(x)  # [B, T, T]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        v = self.proj_v(x)
        x2 = torch.matmul(attn, v)
        x = x + x2
        x = self.norm1(x)
        # feedforward
        x2 = self.ff(x)
        x = x + x2
        x = self.norm2(x)
        return x

class ConvTransForecaster(nn.Module):
    def __init__(
        self,
        enc_input_size: int,
        dec_input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        kernel_size: int = 3,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        super().__init__()
        self.quantiles = quantiles
        self.dim_model = hidden_size
        # initial linear to dim_model
        self.input_proj = nn.Linear(enc_input_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, hidden_size))  # sufficiently long
        self.layers = nn.ModuleList([
            TransformerLayerConv(hidden_size, nhead, kernel_size=kernel_size, dim_feedforward=hidden_size*2)
            for _ in range(num_layers)
        ])
        # decoder: simple linear for each horizon step (direct multi-step)
        self.final_proj = nn.Linear(hidden_size, len(quantiles))

    def forward(self, enc_inp, dec_inp):
        """
        enc_inp: [B, T_enc, Din_enc]
        dec_inp: [B, H, Din_dec]  — we will simply ignore dec_inp for self-attention context,
                                             but could optionally incorporate it.
        returns: [B, H, Q] predictions for each quantile
        """
        B, T_enc, _ = enc_inp.size()
        H = dec_inp.size(1)

        # encode
        x = self.input_proj(enc_inp)  # [B, T_enc, D_model]
        # add positional embedding
        x = x + self.pos_embed[:, :T_enc, :]
        for layer in self.layers:
            x = layer(x, mask=None)  # causal mask optionally could be applied

        # take the last time step representation as "context"
        context = x[:, -1, :]  # [B, D_model]
        # repeat for each horizon
        context_rep = context.unsqueeze(1).repeat(1, H, 1)  # [B, H, D_model]

        # project
        out = self.final_proj(context_rep)  # [B, H, Q]
        return out

def quantile_loss_torch(preds, target, quantiles: List[float]):
    B, H, Q = preds.shape
    target = target.unsqueeze(-1).repeat(1,1,Q)
    losses = []
    for i, q in enumerate(quantiles):
        err = target[..., i] - preds[..., i]
        loss_q = torch.max((q - 1)*err, q*err)
        losses.append(loss_q.unsqueeze(-1))
    loss_all = torch.cat(losses, dim=-1)
    return loss_all.mean()

# ============================================================
# training and evaluation functions
# ============================================================
from tqdm import tqdm, trange

def train_one_epoch(model, loader, optimizer, device, quantiles):
    model.train()
    total_loss = 0.0
    for enc, dec, tgt in tqdm(loader, desc="Training", leave=False):
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
        preds = model(enc, dec)
        loss = quantile_loss_torch(preds, tgt, quantiles)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * enc.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_model(model, loader, device, quantiles):
    model.eval()
    all_tgts = []
    all_preds = {f"p{int(q*100)}": [] for q in quantiles}
    for enc, dec, tgt in loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
        preds = model(enc, dec)
        all_tgts.append(tgt.cpu().numpy())
        for i, q in enumerate(quantiles):
            all_preds[f"p{int(q*100)}"].append(preds[..., i].cpu().numpy())

    targets = np.concatenate(all_tgts, axis=0)
    pred_map = {k: np.concatenate(v, axis=0) for k, v in all_preds.items()}

    # flatten for p50 point metrics
    p50 = pred_map["p50"].reshape(-1)
    target_flat = targets.reshape(-1)
    mask = ~(np.isnan(p50) | np.isnan(target_flat))
    p50 = p50[mask]
    target_flat = target_flat[mask]

    rmse = float(np.sqrt(np.mean((p50 - target_flat)**2)))
    mae  = float(np.mean(np.abs(p50 - target_flat)))
    smape = float(smape_np(target_flat, p50))

    nql = {}
    for q in quantiles:
        qname = f"p{int(q*100)}"
        nql[f"nql_{int(q*100)}"] = float(numpy_normalised_quantile_loss(targets, pred_map[qname], q))

    metrics = {"rmse": rmse, "mae": mae, "smape": smape, "sample_size": int(len(target_flat))}
    return metrics, pred_map, targets, nql

# ============================================================
# main function
# ============================================================
def main():
    csv_path = "path/to/202507-citibike-tft-features-cleaned-500000-isnight.csv"
    df = pd.read_csv(csv_path, parse_dates=["time"])
    print("Loaded data:", df.shape)

    if "sensor_day" not in df.columns:
        df["sensor_day"] = (df["time"].dt.floor("D") - df["time"].dt.floor("D").min()).dt.days

    formatter = CitibikeFormatterLite()
    train_df, valid_df, test_df = formatter.split_data(df, valid_boundary="auto", test_boundary="auto")

    encoder_length = 168
    forecast_horizon = 24

    train_dataset = CitibikeSeqDataset(train_df, formatter, encoder_length, forecast_horizon, split_name="train")
    valid_dataset = CitibikeSeqDataset(valid_df, formatter, encoder_length, forecast_horizon, split_name="valid")
    test_dataset  = CitibikeSeqDataset(test_df,  formatter, encoder_length, forecast_horizon, split_name="test")

    print("Num samples → train:", len(train_dataset), "valid:", len(valid_dataset), "test:", len(test_dataset))

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,  shuffle=False)

    enc_input_size = len(formatter.observed_input_cols + formatter.known_input_cols + formatter.static_input_cols)
    dec_input_size = len(formatter.known_input_cols + formatter.static_input_cols)

    quantiles = [0.1, 0.5, 0.9]
    model = ConvTransForecaster(
        enc_input_size=enc_input_size,
        dec_input_size=dec_input_size,
        hidden_size=128,
        num_layers=2,
        nhead=4,
        kernel_size=3,
        quantiles=quantiles
    )

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float('inf')
    best_state = None
    num_epochs = 10

    for epoch in trange(1, num_epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, quantiles)
        val_metrics, _, _, _ = evaluate_model(model, valid_loader, device, quantiles)
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} valid_RMSE={val_metrics['rmse']:.4f} valid_MAE={val_metrics['mae']:.4f} valid_sMAPE={val_metrics['smape']:.2f}")

        if val_metrics["rmse"] < best_val:
            best_val = val_metrics["rmse"]
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, test_pred_map, test_targets, test_nql = evaluate_model(model, test_loader, device, quantiles)
    print("\n"+"="*50)
    print("TEST POINT FORECAST METRICS (using P50)")
    print("="*50)
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAE:  {test_metrics['mae']:.4f}")
    print(f"sMAPE: {test_metrics['smape']:.2f}%")
    print(f"Sample size: {test_metrics['sample_size']}")
    print("="*50)
    print("Normalized Quantile Loss on test:")
    for k, v in test_nql.items():
        print(f"  {k}: {v:.4f}")

    # save test predictions
    horizon = test_pred_map["p50"].shape[1]
    rows = []
    for i in range(test_pred_map["p50"].shape[0]):
        row = {"sample_id": i}
        for h in range(horizon):
            row[f"p10_t+{h+1}"] = test_pred_map["p10"][i, h]
            row[f"p50_t+{h+1}"] = test_pred_map["p50"][i, h]
            row[f"p90_t+{h+1}"] = test_pred_map["p90"][i, h]
            row[f"y_t+{h+1}"]   = test_targets[i, h]
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_df.to_csv("citibike_convtrans_test_predictions.csv", index=False)
    print("Saved predictions to citibike_convtrans_test_predictions.csv")

if __name__ == "__main__":
    main()
