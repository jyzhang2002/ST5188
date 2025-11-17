import os
import math
from typing import List, Dict, Tuple
import datetime as dte
from datetime import datetime
import argparse
import datetime as dte
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------- logging (same as original) -------------------------
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


log_dir = "lstm_logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
log_file = open(log_path, "w", buffering=1)
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

# ============================================================
# 1. functions
# ============================================================

def numpy_normalised_quantile_loss(y: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    prediction_underflow = y - y_pred
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.0) + (
        1.0 - quantile
    ) * np.maximum(-prediction_underflow, 0.0)
    quantile_loss = weighted_errors.mean()
    normaliser = np.abs(y).mean()
    return 2 * quantile_loss / (normaliser + 1e-8)


def smape_np(y_true, y_pred):
    return np.mean(200.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


# ============================================================
# 2. formatter
# ============================================================

class CitibikeFormatterLite:
    def __init__(self):
        self.identifier_cols = ["station_id", "time"]
        self.target_col = "values"

        # observed
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

        # known future inputs
        self.known_input_cols = [
            "hour", "dow", "is_weekend", "sensor_day", "is_night"
        ]

        # static
        self.static_input_cols = [
            "latitude", "longitude", "station_name", "station_id"
        ]

        # categorical columns
        self.categorical_cols = [
            "station_id", "dow", "is_weekend", "is_night", "station_name"
        ]

        self.scalers = {}
        self.label_encoders = {}

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

        valid_boundary = 22
        test_boundary = 26

        if valid_boundary in (None, 'auto') or test_boundary in (None, 'auto'):
            valid_boundary, test_boundary = self._propose_boundaries(num_days)
            print(f"[CitibikeFormatterLite] Auto boundaries -> valid={valid_boundary}, test={test_boundary}, num_days={num_days}")

        # keep same logic as TFT
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
        test = df.loc[index >= test_boundary - 7]

        # train fit
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
# 3. window Dataset
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

        for sid, sub in self.df.groupby("station_id"):
            sub = sub.sort_values("time")
            sub_idx = sub.index.to_list()
            values_len = len(sub_idx)

            if values_len < total:
                continue

            for local_start in range(0, values_len - total + 1):
                global_start = sub_idx[local_start]
                global_end = sub_idx[local_start + total - 1]

                dec_start_local = local_start + enc 
                dec_end_local = local_start + total 
                dec_rows_global = sub_idx[dec_start_local:dec_end_local]
                window_target = self.df.loc[dec_rows_global, self.target_col]
                if window_target.isna().all():
                    continue

                self.indices.append(global_start)

        print(f"[{self.split_name}] built total windows: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_global = self.indices[idx]
        enc_end = start_global + self.encoder_length
        dec_end = enc_end + self.forecast_horizon

        enc_df = self.df.iloc[start_global:enc_end]
        dec_df = self.df.iloc[enc_end:dec_end]

        static_vals = enc_df[self.fmt.static_input_cols].iloc[0].values.astype(np.float32)
        static_broadcast_enc = np.repeat(static_vals[None, :], self.encoder_length, axis=0)
        static_broadcast_dec = np.repeat(static_vals[None, :], self.forecast_horizon, axis=0)

        enc_feat = enc_df[self.fmt.observed_input_cols + self.fmt.known_input_cols].values.astype(np.float32)
        enc_feat = np.concatenate([enc_feat, static_broadcast_enc], axis=1)

        dec_known = dec_df[self.fmt.known_input_cols].values.astype(np.float32)
        dec_feat = np.concatenate([dec_known, static_broadcast_dec], axis=1)

        target = dec_df[self.target_col].values.astype(np.float32)

        return (
            torch.from_numpy(enc_feat),
            torch.from_numpy(dec_feat),
            torch.from_numpy(target),
        )


# ============================================================
# 4. LSTM model
# ============================================================

class LSTMSeq2SeqQuantile(nn.Module):
    def __init__(
        self,
        enc_input_size: int,
        dec_input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ):
        super().__init__()
        self.quantiles = quantiles
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(
            input_size=enc_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.decoder = nn.LSTM(
            input_size=dec_input_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.proj = nn.Linear(hidden_size, len(quantiles))

    def forward(self, enc_inp, dec_inp):
        enc_out, (h, c) = self.encoder(enc_inp)
        context = h[-1]  # [B, H]

        B, T, Ddec = dec_inp.shape
        context_rep = context.unsqueeze(1).repeat(1, T, 1)
        dec_input_cat = torch.cat([dec_inp, context_rep], dim=-1)
        dec_out, _ = self.decoder(dec_input_cat)
        out = self.proj(dec_out)
        return out


def quantile_loss_torch(preds, target, quantiles: List[float]):
    B, T, Q = preds.shape
    target = target.unsqueeze(-1).repeat(1, 1, Q)

    losses = []
    for qi, q in enumerate(quantiles):
        errors = target[..., qi] - preds[..., qi]
        loss = torch.max((q - 1) * errors, q * errors)
        losses.append(loss.unsqueeze(-1))
    loss_all = torch.cat(losses, dim=-1)
    return loss_all.mean()


# ============================================================
# 5. train & evaluate
# ============================================================
from tqdm import tqdm
from tqdm import trange

def train_one_epoch(model, loader, optimizer, device, quantiles):
    model.train()
    total_loss = 0.0
    
    # tqdm
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch_idx, (enc, dec, tgt) in enumerate(pbar):
        enc = enc.to(device)
        dec = dec.to(device)
        tgt = tgt.to(device)

        preds = model(enc, dec)
        loss = quantile_loss_torch(preds, tgt, quantiles)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * enc.size(0)
    
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_model(model, loader, device, quantiles):
    model.eval()
    all_tgts = []
    all_preds = {f"p{int(q*100)}": [] for q in quantiles}

    for enc, dec, tgt in loader:
        enc = enc.to(device)
        dec = dec.to(device)
        tgt = tgt.to(device)

        preds = model(enc, dec)

        all_tgts.append(tgt.cpu().numpy())
        for qi, q in enumerate(quantiles):
            all_preds[f"p{int(q*100)}"].append(preds[..., qi].cpu().numpy())

    targets = np.concatenate(all_tgts, axis=0)
    process_map = {}
    for k, v in all_preds.items():
        process_map[k] = np.concatenate(v, axis=0)

    p50 = process_map["p50"]
    pred_flat = p50.reshape(-1)
    target_flat = targets.reshape(-1)

    mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask]

    mse = np.mean((pred_flat - target_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_flat - target_flat))
    smape = smape_np(target_flat, pred_flat)

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "smape": float(smape),
        "sample_size": int(len(pred_flat)),
    }

    nqll = {}
    for q in quantiles:
        nqll[f"nql_{int(q*100)}"] = float(
            numpy_normalised_quantile_loss(targets, process_map[f"p{int(q*100)}"], q)
        )

    return metrics, process_map, targets, nqll


# ============================================================
# 6. main
# ============================================================

def main():
    csv_path = "path/to/202507-citibike-tft-features-cleaned-500000-isnight.csv"
    df = pd.read_csv(csv_path, parse_dates=["time"])
    print("Loaded data:", df.shape)

    if "sensor_day" not in df.columns:
        df["sensor_day"] = (df["time"].dt.floor("D") - df["time"].dt.floor("D").min()).dt.days

    formatter = CitibikeFormatterLite()
    train_df, valid_df, test_df = formatter.split_data(df, valid_boundary="auto", test_boundary="auto")

    print("Split sizes:", len(train_df), len(valid_df), len(test_df))

    encoder_length = 168
    forecast_horizon = 24

    train_dataset = CitibikeSeqDataset(train_df, formatter, encoder_length, forecast_horizon, split_name="train")
    valid_dataset = CitibikeSeqDataset(valid_df, formatter, encoder_length, forecast_horizon, split_name="valid")
    test_dataset = CitibikeSeqDataset(test_df, formatter, encoder_length, forecast_horizon, split_name="test")

    print("Num samples -> train:", len(train_dataset), "valid:", len(valid_dataset), "test:", len(test_dataset))

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    enc_input_size = len(formatter.observed_input_cols + formatter.known_input_cols + formatter.static_input_cols)
    dec_input_size = len(formatter.known_input_cols + formatter.static_input_cols)

    quantiles = [0.1, 0.5, 0.9]
    model = LSTMSeq2SeqQuantile(
        enc_input_size=enc_input_size,
        dec_input_size=dec_input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        quantiles=quantiles,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = 1e9
    best_state = None
    num_epochs = 5

    for epoch in trange(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, quantiles)
        val_metrics, _, _, _ = evaluate_model(model, valid_loader, device, quantiles)

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} "
              f"valid_RMSE={val_metrics['rmse']:.4f} valid_MAE={val_metrics['mae']:.4f} valid_sMAPE={val_metrics['smape']:.2f}")

        if val_metrics["rmse"] < best_val:
            best_val = val_metrics["rmse"]
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, test_process_map, test_targets, test_nql = evaluate_model(model, test_loader, device, quantiles)

    print("\n" + "=" * 50)
    print("TEST POINT FORECAST METRICS (using P50)")
    print("=" * 50)
    print(f"RMSE:  {test_metrics['rmse']:.4f}")
    print(f"MAE:   {test_metrics['mae']:.4f}")
    print(f"sMAPE: {test_metrics['smape']:.2f}%")
    print(f"Sample size: {test_metrics['sample_size']}")
    print("=" * 50)

    print("Normalized Quantile Loss on test:")
    for k, v in test_nql.items():
        print(f"  {k}: {v:.4f}")

    # save
    horizon = test_process_map["p50"].shape[1]
    rows = []
    for i in range(test_process_map["p50"].shape[0]):
        row = {"sample_id": i}
        for h in range(horizon):
            row[f"p10_t+{h+1}"] = test_process_map["p10"][i, h]
            row[f"p50_t+{h+1}"] = test_process_map["p50"][i, h]
            row[f"p90_t+{h+1}"] = test_process_map["p90"][i, h]
            row[f"y_t+{h+1}"] = test_targets[i, h]
        rows.append(row)
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv("citibike_lstm_test_predictions.csv", index=False)
    print("Saved predictions to citibike_lstm_test_predictions.csv")


if __name__ == "__main__":
    main()
