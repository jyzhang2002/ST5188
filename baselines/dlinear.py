import os
import math
from typing import List, Dict, Tuple
import datetime as dte
from datetime import datetime
import argparse
import datetime as dte
import os
import sys
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
        test = df.loc[index >= test_boundary - 7]

        # use train fit
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
# 3. ✅ window Dataset
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


log_dir = "dlinear_logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
log_file = open(log_path, "w", buffering=1)
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

# ============================================================
# DLinear
# ============================================================

class DLinearQuantile(nn.Module):
    def __init__(
        self,
        enc_input_size: int,     # encoder 
        dec_input_size: int,     # decoder
        horizon: int,
        hidden_size: int = None,  
        quantiles=[0.1, 0.5, 0.9],
        moving_avg_kernel: int = 25
    ):
        super().__init__()
        self.quantiles = quantiles
        self.horizon = horizon
        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size
        self.moving_avg_kernel = moving_avg_kernel

        # trend and seasonal two Linear
        self.linear_trend = nn.Linear(enc_input_size, horizon * len(quantiles))
        self.linear_seasonal = nn.Linear(enc_input_size, horizon * len(quantiles))

    def moving_average(self, x: torch.Tensor, kernel: int):
        pad = kernel - 1
        x_padded = torch.cat([x[:, :1, :].repeat(1, pad, 1), x], dim=1)
        x_avg = torch.nn.functional.avg_pool1d(
            x_padded.transpose(1,2),
            kernel_size=kernel,
            stride=1,
            padding=0
        ).transpose(1,2)
        return x_avg[:, -x.size(1):, :]

    def forward(self, enc_inp, dec_inp):
        B, T_enc, D_enc = enc_inp.shape
        Q = len(self.quantiles)

        trend = self.moving_average(enc_inp, self.moving_avg_kernel)      # [B, T_enc, D_enc]
        seasonal = enc_inp - trend                                     # [B, T_enc, D_enc]

        last_trend = trend[:, -1, :]      # [B, D_enc]
        last_seasonal = seasonal[:, -1, :] # [B, D_enc]

        out_trend = self.linear_trend(last_trend)         # [B, T_dec*Q]
        out_seasonal = self.linear_seasonal(last_seasonal)  # [B, T_dec*Q]

        out = out_trend + out_seasonal                      # [B, T_dec*Q]
        out = out.view(B, self.horizon, Q)                  # [B, T_dec, Q]

        return out

# ============================================================
# train and eval functions
# ============================================================

def train_one_epoch(model, loader, optimizer, device, quantiles):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    for enc, dec, tgt in pbar:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

        preds = model(enc, dec)
        # preds: [B, T_dec, Q], tgt: [B, T_dec]
        # expand tgt to [B, T_dec, Q]
        target_exp = tgt.unsqueeze(-1).repeat(1, 1, len(quantiles))
        loss = 0
        for qi, q in enumerate(quantiles):
            errors = target_exp[..., qi] - preds[..., qi]
            loss += torch.max((q - 1) * errors, q * errors)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * enc.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_model_dlinear(model, loader, device, quantiles):
    model.eval()
    all_tgts, all_preds = [], {f"p{int(q*100)}": [] for q in quantiles}
    for enc, dec, tgt in loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
        preds = model(enc, dec)
        all_tgts.append(tgt.cpu().numpy())
        for qi, q in enumerate(quantiles):
            all_preds[f"p{int(q*100)}"].append(preds[..., qi].cpu().numpy())

    targets = np.concatenate(all_tgts, axis=0)
    process_map = {k: np.concatenate(v, axis=0) for k, v in all_preds.items()}

    p50 = process_map["p50"]
    pred_flat = p50.reshape(-1)
    target_flat = targets.reshape(-1)
    mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
    pred_flat, target_flat = pred_flat[mask], target_flat[mask]

    mse = np.mean((pred_flat - target_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_flat - target_flat))
    smape = smape_np(target_flat, pred_flat)

    metrics = {"rmse": rmse, "mae": mae, "smape": smape, "sample_size": len(pred_flat)}
    nqll = {f"nql_{int(q*100)}": numpy_normalised_quantile_loss(targets, process_map[f"p{int(q*100)}"], q)
            for q in quantiles}

    return metrics, process_map, targets, nqll

# ============================================================
# main function
# ============================================================

def main():
    csv_path = "path/to/202507-citibike-tft-features-cleaned-500000-isnight.csv"
    df = pd.read_csv(csv_path, parse_dates=["time"])
    if "sensor_day" not in df.columns:
        df["sensor_day"] = (df["time"].dt.floor("D") - df["time"].dt.floor("D").min()).dt.days

    formatter = CitibikeFormatterLite()
    train_df, valid_df, test_df = formatter.split_data(df, valid_boundary="auto", test_boundary="auto")

    encoder_length, forecast_horizon = 168, 24
    train_dataset = CitibikeSeqDataset(train_df, formatter, encoder_length, forecast_horizon, split_name="train")
    valid_dataset = CitibikeSeqDataset(valid_df, formatter, encoder_length, forecast_horizon, split_name="valid")
    test_dataset = CitibikeSeqDataset(test_df, formatter, encoder_length, forecast_horizon, split_name="test")

    print("Num samples → train:", len(train_dataset), "valid:", len(valid_dataset), "test:", len(test_dataset))

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    enc_input_size = len(formatter.observed_input_cols + formatter.known_input_cols + formatter.static_input_cols)
    dec_input_size = len(formatter.known_input_cols + formatter.static_input_cols)

    quantiles = [0.1, 0.5, 0.9]
    model = DLinearQuantile(
        enc_input_size=enc_input_size,
        dec_input_size=dec_input_size,
        horizon=forecast_horizon,
        quantiles=quantiles,
        moving_avg_kernel=25
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = 1e9
    best_state = None
    num_epochs = 5
    for epoch in trange(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, quantiles)
        val_metrics, _, _, _ = evaluate_model_dlinear(model, valid_loader, device, quantiles)
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} valid_RMSE={val_metrics['rmse']:.4f}")

        if val_metrics["rmse"] < best_val:
            best_val = val_metrics["rmse"]
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, test_process_map, test_targets, test_nqll = evaluate_model_dlinear(model, test_loader, device, quantiles)
    print("\n==== TEST METRICS (P50) ====")
    print(f"RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, sMAPE: {test_metrics['smape']:.2f}%")
    for k, v in test_nqll.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
