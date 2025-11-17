#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merging.py — Merge five (or more) CitiBike CSVs into one big table (CSV/Parquet),
and export QC including:
  - Per-file row counts
  - Total rows
  - Column-wise missing value count and percentage (over the merged dataset)

Usage (examples)
---------------
python merging.py --glob "202507-citibike-tripdata_*.csv" --out merged_raw.parquet --qc-out merge_qc.xlsx
python merging.py --files a.csv b.csv c.csv d.csv e.csv --out merged_raw.csv --chunksize 200000 --qc-out merge_qc.xlsx
"""
import argparse, os, glob, sys
from collections import defaultdict, Counter
import pandas as pd

def has_pyarrow() -> bool:
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
        return True
    except Exception:
        return False

def iter_csv_chunks(path, chunksize):
    # low_memory False to avoid dtype guessing fragmentation
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        yield chunk

def write_parquet(files, out_path, chunksize, qc_rows, missing_counter, nonnull_counter, columns_seen):
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    total_rows = 0
    for f in files:
        file_rows = 0
        # Determine column order from header of this file
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        for c in cols:
            columns_seen.add(c)
        for chunk in iter_csv_chunks(f, chunksize):
            # update missing stats before any coercion
            total_rows += len(chunk)
            file_rows += len(chunk)
            # Track non-null count per column (union over all seen)
            for c in chunk.columns:
                nn = chunk[c].notna().sum()
                nonnull_counter[c] += int(nn)
            # Track columns that are entirely missing from this chunk (i.e., absent)
            # We'll account for that later when computing pct using union of columns and total_rows
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            else:
                # Align schema by casting if necessary
                if table.schema != writer.schema:
                    table = table.cast(writer.schema)
            writer.write_table(table)
        qc_rows.append({"file": os.path.basename(f), "rows": int(file_rows)})
    if writer is not None:
        writer.close()
    return total_rows

def write_csv(files, out_path, chunksize, qc_rows, missing_counter, nonnull_counter, columns_seen):
    if os.path.exists(out_path):
        os.remove(out_path)
    wrote_header = False
    total_rows = 0
    # union of columns from first file header will be used for writing;
    # but files may have differing columns; we will outer-align per chunk
    all_cols_order = None

    # First pass to compute union column order
    union = []
    seen = set()
    for f in files:
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        for c in cols:
            if c not in seen:
                union.append(c); seen.add(c)
    all_cols_order = union

    with open(out_path, "w", encoding="utf-8", newline="") as out_f:
        for f in files:
            file_rows = 0
            for chunk in iter_csv_chunks(f, chunksize):
                # align columns (outer join semantics -> missing columns filled with NA)
                chunk = chunk.reindex(columns=all_cols_order)
                # update stats
                total_rows += len(chunk)
                file_rows += len(chunk)
                # non-null per column
                for c in all_cols_order:
                    if c in chunk:
                        nonnull_counter[c] += int(chunk[c].notna().sum())
                # write
                if not wrote_header:
                    chunk.to_csv(out_f, index=False, header=True)
                    wrote_header = True
                else:
                    chunk.to_csv(out_f, index=False, header=False)
            qc_rows.append({"file": os.path.basename(f), "rows": int(file_rows)})
        columns_seen.update(all_cols_order)
    return total_rows

def compute_missingness_df(total_rows, columns_seen, nonnull_counter):
    rows = []
    for c in sorted(columns_seen):
        nn = int(nonnull_counter.get(c, 0))
        miss = total_rows - nn
        pct = (miss / total_rows) if total_rows > 0 else float("nan")
        rows.append({"column": c, "non_null": nn, "missing": miss, "missing_pct": round(pct, 6)})
    return pd.DataFrame(rows).sort_values(["missing_pct","column"], ascending=[False, True])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="*", help="Explicit list of CSV files to merge")
    p.add_argument("--glob", default="202507-citibike-tripdata_*.csv", help="Glob pattern for input CSVs")
    p.add_argument("--out", required=True, help="Output merged file (.csv or .parquet)")
    p.add_argument("--chunksize", type=int, default=200_000, help="Read/write chunk size")
    p.add_argument("--qc-out", default="merge_qc.xlsx", help="QC report path (.xlsx or .csv)")
    return p.parse_args()

def main():
    args = parse_args()
    files = args.files or sorted(glob.glob(args.glob))
    if not files:
        print("No input files matched.", file=sys.stderr)
        sys.exit(2)

    qc_rows = []
    nonnull_counter = Counter()
    columns_seen = set()

    if args.out.lower().endswith(".parquet") and has_pyarrow():
        total = write_parquet(files, args.out, args.chunksize, qc_rows, None, nonnull_counter, columns_seen)
    else:
        total = write_csv(files, args.out, args.chunksize, qc_rows, None, nonnull_counter, columns_seen)

    df_files = pd.DataFrame(qc_rows)
    df_files.loc[len(df_files)] = {"file": "__TOTAL__", "rows": int(total)}

    df_missing = compute_missingness_df(total, columns_seen, nonnull_counter)

    # Write QC
    if args.qc_out.lower().endswith(".xlsx"):
        with pd.ExcelWriter(args.qc_out, engine="xlsxwriter") as w:
            df_files.to_excel(w, index=False, sheet_name="file_row_counts")
            df_missing.to_excel(w, index=False, sheet_name="missingness_all")
    else:
        # CSV: we will write two files with suffixes
        base, ext = os.path.splitext(args.qc_out)
        df_files.to_csv(f"{base}_files.csv", index=False)
        df_missing.to_csv(f"{base}_missing.csv", index=False)
    print(f"[OK] Wrote merged file: {args.out}")
    print(f"[QC] Wrote: {args.qc_out}")

if __name__ == "__main__":
    main()
