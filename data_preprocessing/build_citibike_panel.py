#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_citibike_panel.py — From trip-level raw to dense hour×station panel.
Adds comprehensive QC: export missing value counts & proportions to Excel.

Key sheets in QC workbook:
- 00_snapshot           : time range, hours, stations, expected rows, actual rows
- 10_raw_required       : required raw columns missingness (count, pct) on raw trips
- 20_panel_missingness  : final panel columns missingness
- 30_station_activity   : per-station mean dep/arr and zero-share (for context)

Usage:
python build_citibike_panel.py --raw merged_raw.parquet --out panel_jul2025.csv --qc-out panel_qc.xlsx
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

REQUIRED_COLS = [
    "started_at", "ended_at",
    "start_station_id", "start_station_name", "start_lat", "start_lng",
    "end_station_id", "end_station_name", "end_lat", "end_lng",
]


def to_datetime_floor_hour(s: pd.Series) -> pd.Series:
    """Parse to datetime (naive local) and floor to the hour."""
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    return dt.dt.floor("h")


def read_any(path: str, usecols=None, chunksize=None):
    """
    Read CSV (possibly in chunks) or Parquet. Returns an iterable of DataFrames.
    For Parquet, we just return a single-element list.
    """
    if path.lower().endswith(".parquet"):
        return [pd.read_parquet(path, columns=usecols)]
    else:
        if chunksize:
            return pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)
        else:
            return [pd.read_csv(path, usecols=usecols, low_memory=False)]


def build_panel(
    raw_path: str,
    out_path: str,
    qc_out: str,
    start: str = "2025-07-01 00:00",
    end: str = "2025-07-31 23:00",
    chunksize: int = 400_000,
):
    # QC accumulators for raw required columns
    raw_total_rows = 0
    nonnull_counter = Counter({c: 0 for c in REQUIRED_COLS})

    # Aggregation containers
    dep_frames = []
    arr_frames = []
    meta_frames = []

    sb = pd.Timestamp(start)
    eb = pd.Timestamp(end)

    def process_chunk(df: pd.DataFrame):
        nonlocal raw_total_rows
        raw_total_rows += len(df)

        # track non-null counts for REQUIRED_COLS
        for c in REQUIRED_COLS:
            if c in df.columns:
                nonnull_counter[c] += int(df[c].notna().sum())

        # time floor
        df["_start_time"] = to_datetime_floor_hour(df["started_at"])
        df["_end_time"] = to_datetime_floor_hour(df["ended_at"])

        # filter to July window
        dep = df.loc[df["_start_time"].between(sb, eb, inclusive="both")].copy()
        arr = df.loc[df["_end_time"].between(sb, eb, inclusive="both")].copy()

        # departures: count per hour × start_station_name
        gdep = (
            dep.groupby(["_start_time", "start_station_name"], dropna=False)
            .agg(
                departures=("started_at", "size"),
                latitude=("start_lat", "median"),
                longitude=("start_lng", "median"),
            )
            .reset_index()
            .rename(
                columns={
                    "_start_time": "time",
                    "start_station_name": "station_name",
                }
            )
        )

        # arrivals: count per hour × end_station_name
        garr = (
            arr.groupby(["_end_time", "end_station_name"], dropna=False)
            .agg(
                arrivals=("ended_at", "size"),
                latitude_arr=("end_lat", "median"),
                longitude_arr=("end_lng", "median"),
            )
            .reset_index()
            .rename(
                columns={
                    "_end_time": "time",
                    "end_station_name": "station_name",
                }
            )
        )

        # station meta from start / end
        meta_start = (
            dep[["start_station_name", "start_station_id", "start_lat", "start_lng"]]
            .rename(
                columns={
                    "start_station_name": "station_name",
                    "start_station_id": "station_id",
                    "start_lat": "latitude",
                    "start_lng": "longitude",
                }
            )
            .assign(source="start")
        )
        meta_end = (
            arr[["end_station_name", "end_station_id", "end_lat", "end_lng"]]
            .rename(
                columns={
                    "end_station_name": "station_name",
                    "end_station_id": "station_id",
                    "end_lat": "latitude",
                    "end_lng": "longitude",
                }
            )
            .assign(source="end")
        )

        dep_frames.append(gdep)
        arr_frames.append(garr)
        meta_frames.append(pd.concat([meta_start, meta_end], ignore_index=True))

    # Read raw in chunks (CSV) or single frame (Parquet)
    reader = read_any(
        raw_path,
        chunksize=None if raw_path.lower().endswith(".parquet") else chunksize,
    )
    for chunk in reader:
        process_chunk(chunk)

    # Concatenate aggregated pieces
    if dep_frames:
        all_dep = pd.concat(dep_frames, ignore_index=True)
    else:
        all_dep = pd.DataFrame(
            columns=["time", "station_name", "departures", "latitude", "longitude"]
        )

    if arr_frames:
        all_arr = pd.concat(arr_frames, ignore_index=True)
    else:
        all_arr = pd.DataFrame(
            columns=["time", "station_name", "arrivals", "latitude_arr", "longitude_arr"]
        )

    if meta_frames:
        all_meta = pd.concat(meta_frames, ignore_index=True)
    else:
        all_meta = pd.DataFrame(
            columns=["station_name", "station_id", "latitude", "longitude", "source"]
        )

    # Choose dominant station_id per station_name
    meta_counts = (
        all_meta.dropna(subset=["station_name"])
        .groupby(["station_name", "station_id"], dropna=False)
        .size()
        .reset_index(name="n")
    )

    meta_top = (
        meta_counts.sort_values(["station_name", "n"], ascending=[True, False])
        .drop_duplicates(subset=["station_name"], keep="first")
        .merge(
            all_meta.drop(columns=["source"]).drop_duplicates(),
            how="left",
            on=["station_name", "station_id"],
        )
    )

    meta_top = meta_top[["station_name", "station_id", "latitude", "longitude"]].drop_duplicates()

    # Build dense hours × stations index
    hours = pd.date_range(sb, eb, freq="h")
    stations = meta_top["station_name"].dropna().unique()
    idx = pd.MultiIndex.from_product([hours, stations], names=["time", "station_name"])
    dense = pd.DataFrame(index=idx).reset_index()

    # Merge dep/arr into dense frame
    panel = (
        dense.merge(all_dep, on=["time", "station_name"], how="left")
        .merge(all_arr, on=["time", "station_name"], how="left")
    )

    # Prefer dep lat/lng, fallback to arr lat/lng
    panel["latitude"] = panel["latitude"].fillna(panel["latitude_arr"])
    panel["longitude"] = panel["longitude"].fillna(panel["longitude_arr"])
    panel = panel.drop(columns=["latitude_arr", "longitude_arr"], errors="ignore")

    # Merge station meta to bring in station_id (and possibly better lat/lng)
    # 注意：这里只会产生 latitude_meta / longitude_meta 这类列，
    # 不会有 station_id_meta，所以不再引用 station_id_meta。
    panel = panel.merge(
        meta_top,
        on="station_name",
        how="left",
        suffixes=("", "_meta"),  # latitude_meta / longitude_meta 只来自 meta_top
    )

    # latitude/longitude 再次用 meta 补全
    if "latitude_meta" in panel.columns:
        panel["latitude"] = panel["latitude"].fillna(panel["latitude_meta"])
    if "longitude_meta" in panel.columns:
        panel["longitude"] = panel["longitude"].fillna(panel["longitude_meta"])

    # 清理 *_meta 列
    panel = panel.drop(columns=[c for c in panel.columns if c.endswith("_meta")])

    # Fill NA counts with 0
    panel["arrivals"] = panel["arrivals"].fillna(0).astype("int32")
    panel["departures"] = panel["departures"].fillna(0).astype("int32")

    # Time features
    panel["hour"] = panel["time"].dt.hour.astype("int16")
    panel["dow"] = panel["time"].dt.isocalendar().day.astype("int16")
    panel["is_weekend"] = (panel["dow"] >= 6).astype("int8")

    # Sort and save panel
    panel = panel.sort_values(["time", "station_name"]).reset_index(drop=True)
    panel.to_csv(out_path, index=False)

    # -------- QC exports --------

    # 00 snapshot
    T = len(hours)
    S = len(stations)
    expected = T * S
    actual = len(panel)
    snap = pd.DataFrame(
        [
            {
                "time_start": hours[0],
                "time_end": hours[-1],
                "T_hours": T,
                "S_stations": S,
                "rows_expected": expected,
                "rows_actual": actual,
                "raw_rows_seen": raw_total_rows,
            }
        ]
    )

    # 10 raw required missingness
    raw_missing = []
    for c in REQUIRED_COLS:
        nn = int(nonnull_counter.get(c, 0))
        miss = raw_total_rows - nn
        pct = (miss / raw_total_rows) if raw_total_rows > 0 else float("nan")
        raw_missing.append(
            {
                "column": c,
                "non_null": nn,
                "missing": miss,
                "missing_pct": round(pct, 6),
            }
        )
    df_raw_missing = pd.DataFrame(raw_missing)

    # 20 panel missingness (column-wise)
    def missingness(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        total = len(df)
        for col in df.columns:
            miss = int(df[col].isna().sum())
            pct = miss / total if total > 0 else float("nan")
            rows.append(
                {"column": col, "missing": miss, "missing_pct": round(pct, 6)}
            )
        return pd.DataFrame(rows).sort_values(
            ["missing_pct", "column"], ascending=[False, True]
        )

    df_panel_miss = missingness(panel)

    # 30 station activity summary
    station_activity = (
        panel.groupby("station_name")
        .agg(
            mean_dep=("departures", "mean"),
            mean_arr=("arrivals", "mean"),
            zero_share_dep=("departures", lambda s: (s == 0).mean()),
            zero_share_arr=("arrivals", lambda s: (s == 0).mean()),
        )
        .reset_index()
    )

    # Write QC workbook
    if qc_out.lower().endswith(".xlsx"):
        with pd.ExcelWriter(
            qc_out, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm"
        ) as w:
            snap.to_excel(w, index=False, sheet_name="00_snapshot")
            df_raw_missing.to_excel(
                w, index=False, sheet_name="10_raw_required"
            )
            df_panel_miss.to_excel(
                w, index=False, sheet_name="20_panel_missingness"
            )
            station_activity.to_excel(
                w, index=False, sheet_name="30_station_activity"
            )
    else:
        base, _ = os.path.splitext(qc_out)
        snap.to_csv(f"{base}_snapshot.csv", index=False)
        df_raw_missing.to_csv(f"{base}_raw_required.csv", index=False)
        df_panel_miss.to_csv(f"{base}_panel_missingness.csv", index=False)
        station_activity.to_csv(
            f"{base}_station_activity.csv", index=False
        )

    return panel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="Merged raw trips .csv or .parquet")
    p.add_argument("--out", required=True, help="Output panel CSV path")
    p.add_argument(
        "--qc-out", default="panel_qc.xlsx", help="QC workbook path (.xlsx or .csv)"
    )
    p.add_argument("--start", default="2025-07-01 00:00")
    p.add_argument("--end", default="2025-07-31 23:00")
    p.add_argument("--chunksize", type=int, default=400_000)
    return p.parse_args()


def main():
    args = parse_args()
    build_panel(
        args.raw, args.out, args.qc_out, args.start, args.end, args.chunksize
    )


if __name__ == "__main__":
    main()
