# 🚲 ST5188-TFT4CitiBike

## 📘 Project Overview
This project implements the **Temporal Fusion Transformer (TFT)** model to perform **time series forecasting** on **station-level Citi Bike demand data** in New York City.

Our work is mainly based on the original TFT implementation developed by the Google research team:  
<https://github.com/google-research/google-research/tree/master/tft>. Since the original codebase was written in TensorFlow 1.x, which is incompatible with modern A800 CUDA/GPU environments, we reimplemented the model in PyTorch.

In addition to reproducing the original TFT architecture, **two improvements** are introduced to enhance model performance and interpretability.

---

## 🧠 Objectives
- Model short-term station-level bike demand patterns (hourly or 3-hourly resolution).
- Capture both **temporal dependencies** and **cross-station relationships** using a unified deep learning framework.
- Explore interpretability through **attention weights** and **variable importance**.
- Evaluate model performance with multiple baselines (e.g., LSTM, GRU, DLinear, DeepAR, ConvTrans).

---

## 🧩 Directory Structure

```

TFT4Citibike/
├── baselines/                  
├── data/                       
├── data_formatters/ 
├── data_preprocessing/           
├── expt_settings/              
├── libs/                       
├── logs/                       
├── README.md                            
├── requirements.txt            
├── script_download_data.py     
├── script_hyperparam_opt.py    
└── script_train_fixed_params.py

```

---

## ⚙️ Environment Setup

We recommend using conda or micromamba for environment management.

```bash
# 1️⃣ Create a new environment
conda create -n <env_name> python==3.10

# 2️⃣ Activate environment
conda activate <env_name>

# 3️⃣ Install dependencies
pip install -r requirements.txt
```

---

## 🗃️ Data Preprocessing

If you wish to reproduce the full data-processing pipeline, navigate to `data_preprocessing/` and review the complete workflow there (including an additional `README.md` inside the folder).

You may also directly use the final training-ready dataset provided in the `data/` section.


---

## 🔍 Feature Selection

We directly provide the final results from the feature selection process, which can be found in `citibike.py`.

If you wish to reproduce this step, navigate to:

```bash
cd data/data/citibike/select_features.py
```
 
Since the procedure randomly samples 50000 training instances, the resulting feature list may vary slightly across runs. Alternatively, you may use the predefined features defined in `citibike.py` to replicate the results. Please use a subset of 500,000 samples after data preprocessing.

---

## 📦 Dataset

The dataset is stored under:

```bash
data/data/citibike/
```

Due to GitHub’s 100 MB file-size limit, the final processed datasets **cannot be uploaded to this repository**. The following files are therefore **not included** in the GitHub repo:

The file **`202507-citibike-tft-features-cleaned-500000-isnight.csv`** corresponds to the **1-hour resolution** dataset, which is used for the vanilla TFT and RevIN-TFT models.

The file **`202507-citibike-tft-features-cleaned-500000-isnight-aggre3hours.csv`** is the **3-hour aggregated** version of the dataset.

The processed Citi Bike dataset can be downloaded via Google Drive:

👉 [Download ST5188 Dataset (Google Drive)] (https://drive.google.com/drive/folders/1w7gSu6ahdKWVBb2MWvDSpAm39R5SHU1q?usp=sharing)


---

## 🚀 Training and Evaluation

Run the following command to start TFT model training:

```bash
python script_train_fixed_params.py citibike data yes
````

**Arguments:**

| Argument   | Description                                |
| ---------- | ------------------------------------------ |
| `citibike` | Dataset identifier                         |
| `data`     | Output directory                       |
| `yes`      | Whether to use gpu for computing |

---

## ⚙️ Modifying Hyperparameters

Most model hyperparameters (e.g., learning rate, hidden size, batch size, dropout rate, attention heads) are defined inside the dataset-specific formatter.

To modify them, navigate to:

```bash
cd data_formatters
````

Then open:

```bash
citibike.py
```

---

## 📊 Model Outputs

During and after training, several artifacts are automatically generated to support evaluation, interpretation, and reproducibility. Below is an overview of the key outputs.

### **1. Training Logs (`logs/`)**
Contain detailed training progress, including loss curves, validation metrics, and checkpoint summaries. These logs help track convergence and diagnose potential training issues.

---

### **2. Saved Model (`data/saved_models/citibike/fixed/`)**
This directory stores the trained Temporal Fusion Transformer model:

- **`TemporalFusionTransformer.pt`**  
  The serialized PyTorch checkpoint of the final trained model.

---

### **3. Forecast Predictions (`data/saved_models/citibike/fixed/predictions/`)**
Quantile-based forecasting results generated by the TFT model:

- **`predictions_p10.csv`**
- **`predictions_p50.csv`**
- **`predictions_p90.csv`**

---

### **4. Attention Analysis (`data/saved_models/citibike/fixed/analysis_plots/`)**
Contains interpretability outputs related to the model’s temporal attention:

- **`attention_matrix.csv`**  
  Captures attention weights that show how past time steps influence predictions.  

---

### **5. Feature Importance (`data/saved_models/citibike/fixed/analysis_plots/`)**
Global variable-importance scores obtained from the TFT’s Variable Selection Network:

- **`static_importance.csv`**
- **`historical_importance.csv`**
- **`future_importance.csv`**

---

## 🧩 Model Improvements

This project extends the original TFT with two major improvements:

### **1. Time Aggregation**

To use this feature, first open `citibike.py` and update `get_num_samples_for_calibration` to match the new number of aggregated samples.  

Update the time-step configuration as follows:

```python
'total_time_steps': 8 * 8,
'num_encoder_steps': 7 * 8,
```

Next, navigate to `expt_settings/configs.py` and modify the dataset path:

```python
"citibike": "202507-citibike-tft-features-cleaned-500000-isnight-aggre3hours.csv",
```

### **2. Reversible Instance Normalization (RevIN)**

To enable RevIN, update the model class in `script_train_fixed_params.py`:

```python
ModelClass = libs.tft_model_improve.TemporalFusionTransformer
```

Then simply run:

```bash
python script_train_fixed_params.py citibike data yes
```

This implementation was developed based on the descriptions provided in the original paper.

---

## 🧪 Baseline Models

For comparison, several baselines are implemented under `baselines/`:

| Model          | Description                                                                                     |
| -------------- | ----------------------------------------------------------------------------------------------- |
| **DLinear**    | Decomposes the series into trend and seasonal components, using lightweight linear projections. |
| **GRU / LSTM** | Recurrent neural networks that capture sequential dependencies.                                 |
| **DeepAR**     | Probabilistic autoregressive forecasting.                              |
| **ConvTrans**  | Convolutional Transformer hybrid combining local temporal patterns and long-range attention.    |

To run any baseline model:

```bash
cd baselines
```

Then open `run_all.py` and specify which model scripts you want to execute (e.g., `dlinear.py`, `lstm.py`, etc.).

Once configured, simply run:

```bash
python run_all.py
```

This script will automatically load the dataset, execute all selected baseline models, and save their predictions and metrics for comparison.

The baseline implementations were developed based on the methodological
descriptions provided in the original papers of each model. No external code
was copied, and all referenced papers are properly cited in the project report.


---

## 📈 Evaluation Metrics

The following metrics are used for performance assessment:

| Metric                                       | Description                                                                     |
| -------------------------------------------- | ------------------------------------------------------------------------------- |
| **RMSE**                                     | Root Mean Squared Error.                  |
| **MAE**                                      | Mean Absolute Error.                      |
| **Normalized Quantile Loss (P10, P50, P90)** | Quantifies accuracy of probabilistic forecasts at different quantiles.          |

---



## 🔮 Future Work

Potential extensions include:

1. **Richer known-future inputs**  
   Adding forecasted weather and other external signals to improve predictive foresight.

2. **Decision-oriented outputs**  
   Converting demand forecasts into actionable pickup/drop-off tasks for operations.

3. **Operations dashboard**  
   Visualizing station forecasts and key metrics (e.g., stock-out minutes, rebalancing effort).

4. **Generalization tests**  
   Evaluating performance across different months, seasons, or nearby cities.

5. **Data and fairness improvements**  
   Incorporating finer-grained mobility data and checking model fairness across stations.




