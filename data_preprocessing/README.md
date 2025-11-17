# Citi Bike July 2025 – Preprocessing Pipeline (ST5188)

This repository contains the preprocessing pipeline used to construct an
hourly station-level demand dataset for Citi Bike in New York City (July 2025),
together with external covariates and detailed QC (quality control) reports.

The dataset as a randomly selected subset of 500,000 observations in the final 
panel is used to train a Temporal Fusion Transformer (TFT) model to forecast 
**net bike demand** (departures minus arrivals) at each station for each hour.
---

## 1. Repository Structure

The key scripts and outputs are:

- `merging.py`  
  Merge the 5 raw Citi Bike trip CSVs for July 2025 into a single trip-level table,
  and generate a QC workbook on missing values in the raw trips.

- `build_citibike_panel.py`  
  Convert the trip-level data into a **dense hour × station panel** on July 2025
  (744 hours × ~2.2k stations), and generate a QC workbook on the panel structure
  and missingness.

- `build_features_tft.py`  
  Merge external covariates (weather, traffic, air quality) onto the panel,
  add missingness indicators, rolling coverage, and lag/rolling window features.
  Also generates a QC workbook summarising missingness and imputation results.

Typical outputs:

- `merge_qc.xlsx` – QC for the raw merged trips.  
- `panel_qc.xlsx` – QC for the hour × station panel.  
- `tft_qc.xlsx` – QC for the feature-enriched panel (TFT-ready).  
- `features_tft_jul2025.csv` – final training dataset (hour × station).

You also used `head5_rows.csv` as a snapshot of the **first five rows** of
`features_tft_jul2025.csv` to check the final schema.

---

## 2. Dependencies

All scripts are pure Python and only need common data libraries:

- Python 3.9+ (any recent version is fine)
- `pandas`
- `numpy`
- `xlsxwriter` (for Excel QC export)
- `pyarrow` (optional, for Parquet output in `merging.py`)

You can install them via:

```bash
pip install pandas numpy xlsxwriter pyarrow