#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_features_tft.py (QC enhanced & rolling fix)

- 合并天气/交通/PM2.5 到 CitiBike 小时×站点面板
- 对天气做“按小时-of-day中位数”插补，并输出插补前后缺失对比
- 统计 traffic / PM2.5 的插值与前后向填补数量及比例
- 导出 citywide fallback、特征覆盖率、滞后有效性
- 修复 groupby().rolling() 赋值引发的索引不兼容错误

生成的 QC 工作簿包含（可能会因是否提供外部数据而有缺省）：
00_snapshot
10_weather_before
11_weather_impute
12_panel_missing_before_impute
13_panel_missing_after_impute
20_traffic_meta
21_traffic_fillstats
30_pm25_fillstats
40_fallback_usage
50_feature_coverage
60_lag_validity
"""
import argparse
import os
import numpy as np
import pandas as pd

NY_TZ = "America/New_York"

# ---------- 时间解析与基础工具 ----------
def floor_hour_local(s):
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.floor("h")

def floor_hour_utc_to_nyc(s):
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_convert(NY_TZ).dt.tz_localize(None).dt.floor("h")

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def missingness(df):
    total = len(df)
    rows = [{"column": c,
             "missing": int(df[c].isna().sum()),
             "missing_pct": (float(df[c].isna().mean()) if total > 0 else float("nan"))}
            for c in df.columns]
    return pd.DataFrame(rows).sort_values(["missing_pct","column"], ascending=[False, True])

def add_missing_masks(df, cols):
    for c in cols:
        if c in df.columns:
            df[f"mask_{c}"] = df[c].isna().astype("int8")
    return df

def add_roll_coverage(df, group_cols, cols, windows=(3,6,24)):
    # transform 会自动对齐索引，适合做 coverage
    df = df.sort_values(group_cols+["time"]).reset_index(drop=True)
    g = df.groupby(group_cols)
    for c in cols:
        if c not in df.columns:
            continue
        notna_series = df[c].notna()
        for W in windows:
            df[f"coverage_{c}_roll{W}h"] = g[notna_series.name].transform(
                lambda s: s.rolling(W, min_periods=1).mean()
            )
    return df

# ---------- 外部特征构造 ----------
def build_weather_features(lga_path):
    import pandas as pd
    import numpy as np

    try:
        df = pd.read_csv(lga_path, low_memory=False)
    except Exception:
        df = pd.read_csv(lga_path, low_memory=False, encoding="latin1")
    # 猜时间列（保留 new 版逻辑）
    time_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ["date", "datetime", "time", "hour", "valid", "observationtime", "timestamp"]:
            time_col = c
            break
    if time_col is None:
        for c in df.columns:
            if any(k in c.lower() for k in ["date", "time", "valid", "observation", "hour"]):
                time_col = c
                break
    if time_col is None:
        raise ValueError("Weather file: cannot find a time column.")

    # 对齐到整点（沿用 new 版的 floor_hour_local）
    df["time"] = floor_hour_local(df[time_col])

    # 复制一份给 ISD 解析使用
    lga = df.copy()

    # --------- old 版里的 ISD 工具函数 ---------
    def _isd_first_number(val, scale=1.0, signed=True):
        if pd.isna(val):
            return pd.NA
        s = str(val)
        part = s.split(",")[0].strip()
        if signed and part.startswith("+"):
            part = part[1:]
        # ISD 缺失码
        if part in {"", "9999", "99999", "999", "99", "-9999"}:
            return pd.NA
        try:
            return float(part) / scale
        except Exception:
            return pd.NA

    def _parse_wnd(s):
        if pd.isna(s):
            return (pd.NA, pd.NA)
        parts = str(s).split(",")
        if len(parts) < 5:
            return (pd.NA, pd.NA)
        dir_raw, spd_raw = parts[0].strip(), parts[3].strip()
        try:
            wdir = float(dir_raw) if dir_raw not in {"", "999", "99"} else pd.NA
        except Exception:
            wdir = pd.NA
        try:
            spd_ms = float(spd_raw) / 10.0
        except Exception:
            spd_ms = pd.NA
        return (wdir, spd_ms)

    # --------- 分支 1：完全沿用 old 版 ISD 解析 ---------
    use_isd = set(["TMP", "DEW", "SLP", "VIS", "WND"]).issubset(set(lga.columns))
    if use_isd:
        # 温度/露点：ISD 原始是 1/10 摄氏度
        lga["temp_c"] = lga["TMP"].apply(lambda v: _isd_first_number(v, scale=10.0))
        lga["dewpt_c"] = lga["DEW"].apply(lambda v: _isd_first_number(v, scale=10.0))
        lga["temp_obs"] = lga["temp_c"] * 9 / 5 + 32
        lga["dewpt_obs"] = lga["dewpt_c"] * 9 / 5 + 32

        # 海平面气压和能见度
        lga["slp_obs"] = lga["SLP"].apply(lambda v: _isd_first_number(v, scale=10.0))
        lga["vis_obs"] = lga["VIS"].apply(lambda v: _isd_first_number(v, scale=1.0, signed=False))

        # 风向风速：WND 编码拆成方向 + m/s，再转 mph
        _w = lga["WND"].apply(_parse_wnd)
        lga["wdir_obs"] = _w.apply(lambda t: t[0])
        lga["wspd_obs"] = _w.apply(lambda t: t[1]) * 2.23694  # mph

        # 降水：优先用 AA1–AA4，单位毫米；没有 AA* 再退回 PRCP
        prcp_cols = [c for c in ["AA1", "AA2", "AA3", "AA4"] if c in lga.columns]

        def _aa_mm(val):
            if pd.isna(val):
                return 0.0
            parts = str(val).split(",")
            if len(parts) < 2:
                return 0.0
            raw = parts[1].strip()
            if raw in {"", "9999"}:
                return 0.0
            try:
                return float(raw)
            except Exception:
                return 0.0

        if prcp_cols:
            lga["prcp_obs"] = lga[prcp_cols].apply(
                lambda row: sum(_aa_mm(row[c]) for c in prcp_cols),
                axis=1,
            ) / 25.4  # mm → inch
        elif "PRCP" in lga.columns:
            import re

            def _prcp_in(val):
                s = str(val)
                try:
                    return float(re.findall(r"[-+]?\\d*\\.?\\d+", s)[0])
                except Exception:
                    return pd.NA

            lga["prcp_obs"] = lga["PRCP"].apply(_prcp_in)

        # 按小时聚合：和 old 完全一致
        agg = lga.groupby("time", as_index=False).agg(
            temp_obs=("temp_obs", "mean"),
            dewpt_obs=("dewpt_obs", "mean"),
            wspd_obs=("wspd_obs", "mean"),
            vis_obs=("vis_obs", "mean"),
            slp_obs=("slp_obs", "mean"),
            prcp_obs=("prcp_obs", "sum"),
            wdir_obs=("wdir_obs", "mean"),
        )
        return agg

    # --------- 分支 2：非 ISD 的通用解析（保留 new 原逻辑） ---------
    cmap = {
        "temp_obs": ["HOURLYDRYBULBTEMPF", "HourlyDryBulbTemperature", "TEMP", "TMP", "temperature", "temp", "air_temp_f"],
        "dewpt_obs": ["HourlyDewPointTemperature", "DEW", "dewpt", "dewpoint_f", "dewpoint"],
        "wspd_obs": ["HourlyWindSpeed", "wind_speed", "wspd", "wind_mph", "WSP"],
        "wdir_obs": ["HourlyWindDirection", "wind_dir", "wdir"],
        "vis_obs":  ["HourlyVisibility", "visibility", "vis"],
        "slp_obs":  ["SeaLevelPressure", "SLP", "slp", "sea_level_pressure"],
        "prcp_obs": ["HourlyPrecipitation", "PRCP", "prcp", "precip", "precipitation_inches"],
    }
    out = {"time": df["time"]}
    for tgt, candidates in cmap.items():
        col = None
        for n in candidates:
            if n in df.columns:
                col = n
                break
            for c in df.columns:
                if c.lower() == n.lower():
                    col = c
                    break
            if col:
                break
        if col:
            out[tgt] = safe_num(df[col])  # 使用 new 版已有的 safe_num
        else:
            out[tgt] = pd.Series([np.nan] * len(df))

    w = (
        pd.DataFrame(out)
        .groupby("time", as_index=False)
        .agg(
            {
                "temp_obs": "mean",
                "dewpt_obs": "mean",
                "wspd_obs": "mean",
                "wdir_obs": "mean",
                "vis_obs": "mean",
                "slp_obs": "mean",
                "prcp_obs": "sum",
            }
        )
    )
    return w

def build_traffic_features(traffic_path, full_hours, qc_store):
    try:
        df = pd.read_csv(traffic_path, low_memory=False)
    except Exception:
        df = pd.read_csv(traffic_path, low_memory=False, encoding="latin1")
    time_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ["dataasof","data_as_of","asof","timestamp","recordeddatetime","date","datetime"]:
            time_col = c; break
        if "as of" in lc or "data as of" in lc:
            time_col = c; break
    if time_col is None:
        qc_store["traffic"] = {"note": "missing timestamp column"}
        return pd.DataFrame({"time": full_hours})

    df["time"] = floor_hour_local(df[time_col])
    spd = next((c for c in df.columns if c.lower() in ["speed","avg_speed","mph"]), None)
    tt  = next((c for c in df.columns if c.lower() in ["travel_time","tt","seconds"]), None)
    df["spd"] = safe_num(df[spd]) if spd else np.nan
    df["tt"]  = safe_num(df[tt])  if tt  else np.nan

    agg = df.groupby("time", as_index=True).agg(
        spd_med=("spd","median"),
        spd_p10=("spd", lambda x: pd.to_numeric(x, errors="coerce").quantile(0.10)
                 if pd.to_numeric(x, errors="coerce").notna().any() else np.nan),
        spd_p90=("spd", lambda x: pd.to_numeric(x, errors="coerce").quantile(0.90)
                 if pd.to_numeric(x, errors="coerce").notna().any() else np.nan),
        tt_med=("tt","median"),
    )
    orig_hours = len(agg)
    agg = agg.reindex(full_hours)  # 对齐到面板小时
    total_hours = len(full_hours)

    fillstats = {}
    for col in ["spd_med","spd_p10","spd_p90","tt_med"]:
        s0 = agg[col].copy()
        s1 = s0.interpolate(limit=3, limit_direction="both")
        interp_mask = s0.isna() & s1.notna()
        s2 = s1.fillna(method="ffill", limit=6).fillna(method="bfill", limit=6)
        ffill_mask = s1.isna() & s2.notna()
        fillstats[col] = {
            "missing_before": int(s0.isna().sum()),
            "missing_before_pct": float(s0.isna().mean()) if total_hours>0 else float("nan"),
            "filled_interp_le3h": int(interp_mask.sum()),
            "filled_interp_le3h_pct": float(interp_mask.mean()) if total_hours>0 else float("nan"),
            "filled_ffill_bfill_le6h": int(ffill_mask.sum()),
            "filled_ffill_bfill_le6h_pct": float(ffill_mask.mean()) if total_hours>0 else float("nan"),
            "missing_after": int(s2.isna().sum()),
            "missing_after_pct": float(s2.isna().mean()) if total_hours>0 else float("nan"),
        }
        agg[col] = s2

    qc_store["traffic"] = {"orig_hours": int(orig_hours),
                           "full_hours": int(total_hours),
                           "per_metric": fillstats}
    return agg.reset_index().rename(columns={"index":"time"})

def build_pm25_features(air_path, full_hours, qc_store):
    try:
        df = pd.read_csv(air_path, low_memory=False)
    except Exception:
        df = pd.read_csv(air_path, low_memory=False, encoding="latin1")
    time_col = next((c for c in df.columns if c.lower() in
                     ["observationtimeutc","time_utc","timestamp_utc","datetime_utc","timeutc","utc"]), None)
    val_col  = next((c for c in df.columns if c.lower() in
                     ["pm2_5","pm25","pm 2.5","value","concentration","pm2.5"]), None)
    if time_col is None or val_col is None:
        qc_store["pm25"] = {"note": "missing columns"}
        city = pd.DataFrame({"time": full_hours, "pm25_mean": np.nan})
        return city

    df["time"] = floor_hour_utc_to_nyc(df[time_col])
    df["pm25"] = safe_num(df[val_col])

    city = df.dropna(subset=["time","pm25"]).groupby("time", as_index=True)["pm25"].mean().to_frame("pm25_mean")
    orig_hours = len(city)
    city = city.reindex(full_hours)
    total_hours = len(full_hours)

    pm_before = city["pm25_mean"].copy()
    city["pm25_mean"] = city["pm25_mean"].interpolate(limit=3, limit_direction="both")
    filled_interp = pm_before.isna() & city["pm25_mean"].notna()
    pm_before2 = city["pm25_mean"].copy()
    city["pm25_mean"] = city["pm25_mean"].fillna(method="ffill", limit=6).fillna(method="bfill", limit=6)
    filled_ffill = pm_before2.isna() & city["pm25_mean"].notna()

    qc_store["pm25"] = {
        "orig_hours": int(orig_hours),
        "full_hours": int(total_hours),
        "missing_before": int(pm_before.isna().sum()),
        "missing_before_pct": float(pm_before.isna().mean()) if total_hours>0 else float("nan"),
        "filled_interp_le3h": int(filled_interp.sum()),
        "filled_interp_le3h_pct": float(filled_interp.mean()) if total_hours>0 else float("nan"),
        "filled_ffill_bfill_le6h": int(filled_ffill.sum()),
        "filled_ffill_bfill_le6h_pct": float(filled_ffill.mean()) if total_hours>0 else float("nan"),
        "missing_after": int(city["pm25_mean"].isna().sum()),
        "missing_after_pct": float(city["pm25_mean"].isna().mean()) if total_hours>0 else float("nan"),
    }
    return city.reset_index().rename(columns={"index":"time"})

# ---------- 派生与滞后/滚动 ----------
def recompute_rh_and_wind(df):
    if "temp_obs" in df.columns and "dewpt_obs" in df.columns:
        T = (df["temp_obs"] - 32) * 5/9
        Td = (df["dewpt_obs"] - 32) * 5/9
        a, b = 17.625, 243.04
        es = 6.1094 * np.exp(a*T/(b+T))
        e  = 6.1094 * np.exp(a*Td/(b+Td))
        df["rh_obs"] = (100 * (e/es)).clip(0, 100)
    if "wdir_obs" in df.columns:
        ang = np.deg2rad(df["wdir_obs"])
        df["wdir_sin_obs"] = np.sin(ang)
        df["wdir_cos_obs"] = np.cos(ang)
    return df

def add_lags_and_rolls(df, group_cols, cols, lags=(1,3,6,24), rolls=(3,6,24)):
    """
    重要修复：不再直接用 groupby().rolling() 赋值，改为 apply(...).reset_index(drop=True)
    以避免 MultiIndex 产物和 DataFrame 索引不兼容。
    """
    # 统一 dtype & 排序，确保索引干净
    for gc in group_cols:
        df[gc] = df[gc].astype(str)
    df = df.sort_values(group_cols+["time"]).reset_index(drop=True)

    g = df.groupby(group_cols, group_keys=False)
    for c in cols:
        if c not in df.columns:
            continue
        # lags
        for L in lags:
            df[f"{c}_lag{L}"] = g[c].shift(L).reset_index(drop=True)

        # rolling features via apply, then hard reset_index(drop=True) to align
        for W in rolls:
            r_mean = g[c].apply(lambda s: s.rolling(W, min_periods=1).mean()).reset_index(drop=True)
            r_sum  = g[c].apply(lambda s: s.rolling(W, min_periods=1).sum()).reset_index(drop=True)
            df[f"{c}_roll{W}h_mean"] = r_mean
            df[f"{c}_roll{W}h_sum"]  = r_sum
    return df

# ---------- 主流程 ----------
def build(panel_path, out_path, qc_out, lga_path=None, traffic_path=None, air_path=None):
    qc_sheets = {}

    panel = pd.read_csv(panel_path, low_memory=False, parse_dates=["time"])
    # station_id 统一成字符串，避免后续分组滚动时 dtype 冲突
    if "station_id" in panel.columns:
        panel["station_id"] = panel["station_id"].astype(str)

    hours = sorted(panel["time"].dropna().unique())
    snap = pd.DataFrame([{
        "rows": len(panel),
        "T_hours": len(hours),
        "S_stations": panel["station_id"].nunique() if "station_id" in panel.columns else panel["station_name"].nunique()
    }])
    qc_sheets["00_snapshot"] = snap

    # 天气：merge 前缺失
    miss_weather_before = {}
    if lga_path:
        w = build_weather_features(lga_path)
        # 记录天气聚合表自身的缺失（插补前）
        for col in ["temp_obs","dewpt_obs","wspd_obs","wdir_obs","vis_obs","slp_obs","prcp_obs"]:
            if col in w.columns:
                miss_weather_before[col] = int(w[col].isna().sum())
        panel = panel.merge(w, on="time", how="left")
        if miss_weather_before:
            qc_sheets["10_weather_before"] = pd.DataFrame.from_dict(
                miss_weather_before, orient="index", columns=["missing_before_merge"]
            )

    # 交通（citywide→扩展到站点）
    qc_store = {}
    if traffic_path:
        traf_city = build_traffic_features(traffic_path, hours, qc_store)
        st = panel[["station_id"]].drop_duplicates().copy()
        st["key"]=1; traf_city["key"]=1
        traf = st.merge(traf_city, on="key").drop(columns=["key"])
        panel = panel.merge(traf, on=["time","station_id"], how="left")

    # PM2.5（citywide→扩展到站点）
    if air_path:
        pm25_city = build_pm25_features(air_path, hours, qc_store)
        st = panel[["station_id"]].drop_duplicates().copy()
        st["key"]=1; pm25_city["key"]=1
        pm25 = st.merge(pm25_city, on="key").drop(columns=["key"])
        panel = panel.merge(pm25, on=["time","station_id"], how="left")

    # ===== 插补前整表缺失（对比用） =====
    qc_sheets["12_panel_missing_before_impute"] = missingness(panel)

    # 按小时-of-day 中位数进行天气插补，并记录插补前后差异
    impute_stats = {}
    hod = panel["time"].dt.hour
    total_rows = len(panel)
    for c in ["temp_obs","dewpt_obs","wspd_obs","wdir_obs","vis_obs","slp_obs","prcp_obs"]:
        if c in panel.columns:
            before_na = int(panel[c].isna().sum())
            med_hod = panel.groupby(hod)[c].transform(lambda s: s.median(skipna=True))
            filled_mask = panel[c].isna() & med_hod.notna()
            panel[c] = panel[c].fillna(med_hod)
            after_na = int(panel[c].isna().sum())
            imputed = int(filled_mask.sum())
            impute_stats[c] = {
                "missing_before": before_na,
                "missing_before_pct": round(before_na/total_rows, 6) if total_rows>0 else np.nan,
                "imputed_hod": imputed,
                "imputed_hod_pct": round(imputed/total_rows, 6) if total_rows>0 else np.nan,
                "missing_after": after_na,
                "missing_after_pct": round(after_na/total_rows, 6) if total_rows>0 else np.nan,
            }
    if impute_stats:
        qc_sheets["11_weather_impute"] = pd.DataFrame(impute_stats).T

    # 衍生 RH 与风向正余弦
    panel = recompute_rh_and_wind(panel)

    # ===== 插补后整表缺失（对比用） =====
    qc_sheets["13_panel_missing_after_impute"] = missingness(panel)

    # 缺失 mask
    cont = [c for c in ["temp_obs","dewpt_obs","rh_obs","wspd_obs","wdir_obs","wdir_sin_obs","wdir_cos_obs",
                        "vis_obs","slp_obs","prcp_obs","spd_med","spd_p10","spd_p90","tt_med","pm25_mean"]
            if c in panel.columns]
    panel = add_missing_masks(panel, cont)

    # 方便后续建模：构造目标 net_demand
    if {"departures","arrivals"}.issubset(panel.columns) and "net_demand" not in panel.columns:
        panel["net_demand"] = panel["departures"] - panel["arrivals"]

    # rolling coverage（按站点）
    panel = add_roll_coverage(panel, ["station_id"], cont, windows=(3,6,24))

    # Lags & rolling（按站点）——这里用修复后的实现，避免索引不兼容
    obs_cols = [c for c in ["pm25_mean","spd_med","prcp_obs","temp_obs","wspd_obs","net_demand"] if c in panel.columns]
    if obs_cols:
        panel = add_lags_and_rolls(panel, ["station_id"], obs_cols, lags=(1,3,6,24), rolls=(3,6,24))

    # fallback 使用统计
    fallback = {}
    for col in ["traffic_citywide_fallback","pm25_citywide_fallback"]:
        if col in panel.columns:
            s = panel[col]
            fallback[col] = {"rows_1": int((s==1).sum()), "rows_total": len(s)}
    if fallback:
        qc_sheets["40_fallback_usage"] = pd.DataFrame(fallback).T

    # 特征覆盖与 lag 有效性
    feat_cov_rows = []
    for c in ["temp_obs","dewpt_obs","rh_obs","wspd_obs","wdir_sin_obs","wdir_cos_obs",
              "vis_obs","slp_obs","prcp_obs","spd_med","spd_p10","spd_p90","tt_med","pm25_mean"]:
        if c in panel.columns:
            fill_rate = float(panel[c].notna().mean())
            g = panel.sort_values(["station_id","time"]).groupby("station_id")[c].apply(
                lambda s: s.notna().rolling(24, min_periods=1).mean())
            cov24_med = float(g.groupby(level=0).median().median())
            feat_cov_rows.append({"feature": c, "fill_rate": fill_rate, "roll24_cov_median": cov24_med})
    if feat_cov_rows:
        qc_sheets["50_feature_coverage"] = pd.DataFrame(feat_cov_rows)

    lag_rows = []
    for base in [c for c in ["pm25_mean","spd_med","prcp_obs","temp_obs","wspd_obs","net_demand"] if c in panel.columns]:
        for L in [1,3,6,24]:
            col = f"{base}_lag{L}"
            if col in panel.columns:
                lag_rows.append({"feature": col, "valid_fraction": float(panel[col].notna().mean())})
    if lag_rows:
        qc_sheets["60_lag_validity"] = pd.DataFrame(lag_rows)

    # traffic / pm25 的填补统计
    if "traffic" in qc_store:
        traf = qc_store["traffic"]
        meta = pd.DataFrame([{"orig_hours": traf.get("orig_hours", np.nan),
                              "full_hours": traf.get("full_hours", np.nan)}])
        qc_sheets["20_traffic_meta"] = meta
        qc_sheets["21_traffic_fillstats"] = pd.DataFrame(traf.get("per_metric", {})).T
    if "pm25" in qc_store:
        qc_sheets["30_pm25_fillstats"] = pd.DataFrame([qc_store["pm25"]])

    # 写出 QC
    if qc_out.lower().endswith(".xlsx"):
        with pd.ExcelWriter(qc_out, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm") as w:
            for name, df in qc_sheets.items():
                df.to_excel(w, index=False, sheet_name=name[:31])
    else:
        base,_ = os.path.splitext(qc_out)
        for name, df in qc_sheets.items():
            df.to_csv(f"{base}_{name}.csv", index=False)

    # 导出特征表
    panel.to_csv(out_path, index=False)
    return panel

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True, help="Input panel CSV")
    p.add_argument("--out", required=True, help="Output enriched features CSV")
    p.add_argument("--lga", dest="lga_path", default=None, help="Weather CSV")
    p.add_argument("--traffic", dest="traffic_path", default=None, help="Traffic speeds CSV")
    p.add_argument("--air", dest="air_path", default=None, help="Air quality (PM2.5) CSV")
    p.add_argument("--qc-out", default="tft_preproc_report.xlsx", help="QC workbook path (.xlsx or .csv)")
    return p.parse_args()

def main():
    args = parse_args()
    build(args.panel, args.out, args.qc_out, args.lga_path, args.traffic_path, args.air_path)

if __name__ == "__main__":
    main()
