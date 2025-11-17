import os
import sys
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------- logging -------------------------
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

log_dir = "deepar_logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
log_file = open(log_path, "w", buffering=1)
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

# ============================================================
# 1) helper functions
# ============================================================
def numpy_normalised_quantile_loss(y: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y - y_pred
    werr = q * np.maximum(diff, 0.0) + (1.0 - q) * np.maximum(-diff, 0.0)
    ql = werr.mean()
    normaliser = np.abs(y).mean()
    return 2 * ql / (normaliser + 1e-8)

def smape_np(y_true, y_pred):
    return np.mean(200.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

# ============================================================
# 2) data formatter
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
            "wspd_obs_roll6h_sum", "pm25_mean_roll6h_mean", "pm25_mean_lag6",
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
            test_b = max(5, num_days - 2)
        else:
            valid_b = int(round(num_days * 0.7))
            test_b = int(round(num_days * 0.85))
        valid_b = max(3, min(valid_b, num_days - 4))
        test_b = max(valid_b + 1, min(num_days - 1, num_days))
        return int(valid_b), int(test_b)

    def split_data(self, df, valid_boundary=None, test_boundary=None):
        print('Formatting train-valid-test splits for full Citi Bike dataset.')
        df = self._ensure_core_columns(df)
        index = df['sensor_day']
        num_days = int(index.max() - index.min() + 1)

        # Default boundaries
        valid_boundary = 22
        test_boundary = 26

        if valid_boundary in (None, 'auto') or test_boundary in (None, 'auto'):
            valid_boundary, test_boundary = self._propose_boundaries(num_days)
            print(f"[CitibikeFormatterLite] Auto boundaries -> valid={valid_boundary}, test={test_boundary}, num_days={num_days}")

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
            df[c] = scaler.transform(v.values.reshape(-1, 1))
        for c, le in self.label_encoders.items():
            v = df[c].astype(str).fillna("__MISSING__")
            v = v.map(lambda x: x if x in le.classes_ else "__MISSING__")
            if "__MISSING__" not in le.classes_:
                le.classes_ = np.append(le.classes_, "__MISSING__")
            df[c] = le.transform(v)
        return df

# ============================================================
# 3) windowed dataset
# ============================================================
class CitibikeSeqDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        formatter: CitibikeFormatterLite,
        encoder_length: int = 168,
        forecast_horizon: int = 24,
        split_name: str = "train",
    ):
        self.df = df.sort_values(["station_id", "time"]).reset_index(drop=True)
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
        static_enc = np.repeat(static_vals[None, :], self.encoder_length, axis=0)
        static_dec = np.repeat(static_vals[None, :], self.forecast_horizon, axis=0)

        enc_feat = enc_df[self.fmt.observed_input_cols + self.fmt.known_input_cols].values.astype(np.float32)
        enc_feat = np.concatenate([enc_feat, static_enc], axis=1)

        dec_known = dec_df[self.fmt.known_input_cols].values.astype(np.float32)
        dec_feat = np.concatenate([dec_known, static_dec], axis=1)

        target = dec_df[self.target_col].values.astype(np.float32)  # [H]

        return (
            torch.from_numpy(enc_feat),  # [enc_len, Din_enc]
            torch.from_numpy(dec_feat),  # [H, Din_dec]
            torch.from_numpy(target),    # [H]
        )

# ============================================================
# 4) DeepAR model
# ============================================================
class DeepAR(nn.Module):
    """
    Encoder: LSTM over encoder covariates
    Decoder: step-wise autoregressive; input = [dec_cov_t, context, y_{t-1}]
    Output: Gaussian mean & scale at each horizon step
    """
    def __init__(
        self,
        enc_input_size: int,
        dec_input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(
            input_size=enc_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder_cell = nn.LSTMCell(
            input_size=dec_input_size + hidden_size + 1,  # dec_cov + context + prev_y
            hidden_size=hidden_size,
        )
        self.mu = nn.Linear(hidden_size, 1)
        self.pre_sigma = nn.Linear(hidden_size, 1)

    def forward(self, enc_inp, dec_inp, tgt=None, teacher_forcing=True):
        B, H, _ = dec_inp.size()

        enc_out, (h, c) = self.encoder(enc_inp)  # h: [L, B, H]
        context = h[-1]                           # [B, H]

        hx = torch.zeros(B, self.hidden_size, device=enc_inp.device)
        cx = torch.zeros(B, self.hidden_size, device=enc_inp.device)

        mus = []
        sigmas = []

        y_prev = torch.zeros(B, 1, device=enc_inp.device)

        for t in range(H):
            dec_cov_t = dec_inp[:, t, :]  # [B, Din_dec]
            dec_step_in = torch.cat([dec_cov_t, context, y_prev], dim=-1)  # [B, Din_dec + H + 1]
            hx, cx = self.decoder_cell(dec_step_in, (hx, cx))
            mu_t = self.mu(hx)                          # [B, 1]
            sigma_t = torch.nn.functional.softplus(self.pre_sigma(hx)) + 1e-6  # [B, 1]
            mus.append(mu_t)
            sigmas.append(sigma_t)

            if teacher_forcing and (tgt is not None):
                y_prev = tgt[:, t].unsqueeze(-1)  # teacher forcing
            else:
                y_prev = mu_t.detach() 

        mu = torch.cat(mus, dim=1)       # [B, H]
        sigma = torch.cat(sigmas, dim=1) # [B, H]
        return mu, sigma

def gaussian_nll(mu, sigma, y):
    inv_var = 1.0 / (sigma ** 2 + 1e-12)
    nll = 0.5 * ((y - mu) ** 2 * inv_var + 2.0 * torch.log(sigma + 1e-12) + np.log(2.0 * np.pi))
    return nll.mean()

def gaussian_quantiles(mu, sigma, quantiles: List[float]):
    out = {}
    std_norm = torch.distributions.Normal(loc=0.0, scale=1.0)
    for q in quantiles:
        z = std_norm.icdf(torch.tensor(q, device=mu.device))
        out[f"p{int(q*100)}"] = mu + sigma * z
    return out

# ============================================================
# 5) training & evaluation functions
# ============================================================
from tqdm import tqdm, trange

@torch.no_grad()
def evaluate_model(model, loader, device, quantiles, scaler_target: StandardScaler):
    model.eval()
    all_targets = []
    all_pdict = {f"p{int(q*100)}": [] for q in quantiles}

    for enc, dec, tgt in loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

        # teacher_forcing=True
        mu, sigma = model(enc, dec, tgt=tgt, teacher_forcing=True)
        qmap = gaussian_quantiles(mu, sigma, quantiles)

        all_targets.append(tgt.cpu().numpy())
        for k, tens in qmap.items():
            all_pdict[k].append(tens.detach().cpu().numpy())

    targets = np.concatenate(all_targets, axis=0)
    pred_map = {k: np.concatenate(v, axis=0) for k, v in all_pdict.items()}

    def inv(x2d):
        x_flat = x2d.reshape(-1, 1)
        y_flat = scaler_target.inverse_transform(x_flat).reshape(x2d.shape)
        return y_flat

    targets_real = inv(targets)
    p50_real = inv(pred_map["p50"])

    mask = ~(np.isnan(p50_real) | np.isnan(targets_real))
    mse = np.mean((p50_real[mask] - targets_real[mask]) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(p50_real[mask] - targets_real[mask])))
    smape = float(smape_np(targets_real[mask], p50_real[mask]))

    nqll = {}
    for q in quantiles:
        qname = f"p{int(q*100)}"
        nqll[f"nql_{int(q*100)}"] = float(
            numpy_normalised_quantile_loss(targets_real, inv(pred_map[qname]), q)
        )

    return {"rmse": rmse, "mae": mae, "smape": smape, "sample_size": int(mask.sum())}, pred_map, targets, nqll

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for enc, dec, tgt in tqdm(loader, desc="Training", leave=False):
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
        mu, sigma = model(enc, dec, tgt=tgt, teacher_forcing=True)
        loss = gaussian_nll(mu, sigma, tgt)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total += loss.item() * enc.size(0)
    return total / len(loader.dataset)

# ============================================================
# 6) main function
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

    print("Num samples -> train:", len(train_dataset), "valid:", len(valid_dataset), "test:", len(test_dataset))

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,  shuffle=False)

    enc_input_size = len(formatter.observed_input_cols + formatter.known_input_cols + formatter.static_input_cols)
    dec_input_size = len(formatter.known_input_cols + formatter.static_input_cols)

    quantiles = [0.1, 0.5, 0.9]
    model = DeepAR(
        enc_input_size=enc_input_size,
        dec_input_size=dec_input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
    )

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = 1e9
    best_state = None
    num_epochs = 10

    from tqdm import trange
    for epoch in trange(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics, _, _, _ = evaluate_model(model, valid_loader, device, quantiles, formatter.scalers[formatter.target_col])

        print(f"[Epoch {epoch:02d}] train_nll={train_loss:.4f} "
              f"valid_RMSE={val_metrics['rmse']:.4f} valid_MAE={val_metrics['mae']:.4f} valid_sMAPE={val_metrics['smape']:.2f}%")

        if val_metrics["rmse"] < best_val:
            best_val = val_metrics["rmse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, test_pmap, test_targets, test_nql = evaluate_model(
        model, test_loader, device, quantiles, formatter.scalers[formatter.target_col]
    )

    print("\n" + "=" * 50)
    print("TEST POINT FORECAST METRICS (using P50, de-standardized)")
    print("=" * 50)
    print(f"RMSE:  {test_metrics['rmse']:.4f}")
    print(f"MAE:   {test_metrics['mae']:.4f}")
    print(f"sMAPE: {test_metrics['smape']:.2f}%")
    print(f"Sample size: {test_metrics['sample_size']}")
    print("=" * 50)

    print("Normalized Quantile Loss on test (de-standardized):")
    for k, v in test_nql.items():
        print(f"  {k}: {v:.4f}")

    scaler_target = formatter.scalers[formatter.target_col]
    def inv_np(x):
        return scaler_target.inverse_transform(x.reshape(-1, 1)).reshape(x.shape)

    horizon = test_pmap["p50"].shape[1]
    rows = []
    p10 = inv_np(test_pmap["p10"])
    p50 = inv_np(test_pmap["p50"])
    p90 = inv_np(test_pmap["p90"])
    y   = inv_np(test_targets)

    for i in range(p50.shape[0]):
        row = {"sample_id": i}
        for h in range(horizon):
            row[f"p10_t+{h+1}"] = p10[i, h]
            row[f"p50_t+{h+1}"] = p50[i, h]
            row[f"p90_t+{h+1}"] = p90[i, h]
            row[f"y_t+{h+1}"]   = y[i, h]
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv("citibike_deepar_test_predictions.csv", index=False)
    print("Saved predictions to citibike_deepar_test_predictions.csv")

if __name__ == "__main__":
    main()
