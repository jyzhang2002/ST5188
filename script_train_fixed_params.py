import argparse
import datetime as dte
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model_original
import libs.tft_model_improve
import libs.utils as utils

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model_original.TemporalFusionTransformer


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


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
log_file = open(log_path, "w", buffering=1)
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)


# ------------------------- small helpers -------------------------
def _to_python_scalars(d):
    """Turn numpy scalars into plain python scalars (int/float/str)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.generic,)):
            out[k] = v.item()
        else:
            out[k] = v
    return out

import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

# set seed for reproducibility
set_seed(42)

def main(expt_name,
         use_gpu,
         model_folder,
         data_csv_path,
         data_formatter,
         use_testing_mode=False):

    if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
        raise ValueError(
            "Data formatter must inherit from GenericDataFormatter, got {}".format(
                type(data_formatter))
        )

    if use_gpu and torch.cuda.is_available():
        # gpu
        torch.cuda.set_device(5) 
        device = "cuda:5"
    else:
        device = "cpu"
    print(f"*** Training from defined parameters for {expt_name} (device={device}) ***")

    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path)
    train, valid, test = data_formatter.split_data(raw_data)
    print(f"Train set size: {len(train)} rows")
    print(f"Valid set size: {len(valid)} rows")
    print(f"Test set size: {len(test)} rows")
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

    fixed_params = data_formatter.get_experiment_params()
    print("fixed_params (before adding column_definition and model_folder): ", fixed_params)
    if "column_definition" not in fixed_params or not isinstance(
        fixed_params["column_definition"], (list, tuple)
    ):
        fixed_params["column_definition"] = data_formatter.get_column_definition()

    fixed_params["model_folder"] = model_folder

    default_model_params = data_formatter.get_default_model_params()
    param_ranges = {k: [default_model_params[k]] for k in default_model_params}

    # test mode
    if use_testing_mode:
        fixed_params["num_epochs"] = 1
        default_model_params["hidden_layer_size"] = 5
        train_samples, valid_samples = 100, 10

    # 3) hyperparam manager
    print("*** Loading hyperparam manager ***")
    opt_manager = HyperparamOptManager(
        param_ranges=param_ranges,
        fixed_params=fixed_params,
        model_folder=model_folder,
    )

    print("*** Running calibration ***")
    print("Training with params (fixed + one choice from ranges):")
    print("fixed_params =")
    for k, v in fixed_params.items():
        print("  {}: {}".format(k, v))
    print("searchable_params =")
    for k, v in param_ranges.items():
        print("  {}: {}".format(k, v))

    best_loss = np.inf

    params = opt_manager.get_next_parameters()
    params = _to_python_scalars(params)

    print(">>> Final merged params sent to model:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    model = ModelClass(params, use_cudnn=use_gpu, device=device)

    if not model.training_data_cached():
        model.cache_batched_data(train, "train", num_samples=train_samples)
        model.cache_batched_data(valid, "valid", num_samples=valid_samples)

    model.fit()
    val_loss = model.evaluate(valid)

    if val_loss < best_loss:
        opt_manager.update_score(params, val_loss, model)
        best_loss = val_loss

    print("*** Running tests ***")
    best_params = opt_manager.get_best_params()
    best_params = _to_python_scalars(best_params)
    best_params["column_definition"] = data_formatter.get_column_definition()

    best_model = ModelClass(best_params, use_cudnn=use_gpu, device=device)
    best_model.load(opt_manager.hyperparam_folder)

    print("Computing best validation loss again...")
    val_loss = best_model.evaluate(valid)

    print(f"Number of rows in test DataFrame: {len(test)}")
    print("Computing test loss...")
    output_map = best_model.predict(test, return_targets=True)

    targets = data_formatter.format_predictions(output_map["targets"])
    p10_forecast = data_formatter.format_predictions(output_map["p10"])
    p50_forecast = data_formatter.format_predictions(output_map["p50"])
    p90_forecast = data_formatter.format_predictions(output_map["p90"])

    # ============================================================
    # Helper
    # ============================================================
    def extract_numerical_data(df):
        return df[[c for c in df.columns if c not in {"forecast_time", "identifier"}]]

    def compute_basic_metrics(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        smape = 100 * np.mean(
            2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
        )
        return rmse, mae, smape

    def quantile_loss(y_true, y_pred, q):
        e = y_true - y_pred
        return np.mean(np.maximum(q * e, (q - 1) * e))

    y_true_all = extract_numerical_data(targets).values.flatten()
    y_p10_all = extract_numerical_data(p10_forecast).values.flatten()
    y_p50_all = extract_numerical_data(p50_forecast).values.flatten()
    y_p90_all = extract_numerical_data(p90_forecast).values.flatten()

    valid_mask = ~np.isnan(y_true_all)
    y_true_all = y_true_all[valid_mask]
    y_p10_all = y_p10_all[valid_mask]
    y_p50_all = y_p50_all[valid_mask]
    y_p90_all = y_p90_all[valid_mask]

    rmse_all, mae_all, smape_all = compute_basic_metrics(y_true_all, y_p50_all)
    p10_loss_all = utils.numpy_normalised_quantile_loss(y_true_all, y_p10_all, 0.1)
    p50_loss_all = utils.numpy_normalised_quantile_loss(y_true_all, y_p50_all, 0.5)
    p90_loss_all = utils.numpy_normalised_quantile_loss(y_true_all, y_p90_all, 0.9)

    eps = 1e-6
    mask_nonzero = np.abs(y_true_all) > eps

    y_true_nz = y_true_all[mask_nonzero]
    y_p10_nz = y_p10_all[mask_nonzero]
    y_p50_nz = y_p50_all[mask_nonzero]
    y_p90_nz = y_p90_all[mask_nonzero]

    if len(y_true_nz) == 0:
        print("⚠️ Warning: after filtering non-zero targets, no samples left. "
              "You may need to increase the threshold (eps).")
        rmse_nz = mae_nz = smape_nz = float("nan")
        p10_loss_nz = p50_loss_nz = p90_loss_nz = float("nan")
    else:
        rmse_nz, mae_nz, smape_nz = compute_basic_metrics(y_true_nz, y_p50_nz)
        p10_loss_nz = utils.numpy_normalised_quantile_loss(y_true_nz, y_p10_nz, 0.1)
        p50_loss_nz = utils.numpy_normalised_quantile_loss(y_true_nz, y_p50_nz, 0.5)
        p90_loss_nz = utils.numpy_normalised_quantile_loss(y_true_nz, y_p90_nz, 0.9)

    print("\n==================================================")
    print("📊 EVALUATION ON ALL SAMPLES")
    print("RMSE: {:.4f} | MAE: {:.4f} | Sample size: {}".format(
        rmse_all, mae_all, len(y_true_all)))
    print("Normalised Quantile Loss: P10={:.4f}, P50={:.4f}, P90={:.4f}".format(
        p10_loss_all, p50_loss_all, p90_loss_all))

    print("--------------------------------------------------")
    print("📈 EVALUATION ON NON-ZERO TARGETS (|y| > {})".format(eps))
    print("RMSE: {:.4f} | MAE: {:.4f} | Sample size: {}".format(
        rmse_nz, mae_nz, len(y_true_nz)))
    print("Normalised Quantile Loss: P10={:.4f}, P50={:.4f}, P90={:.4f}".format(
        p10_loss_nz, p50_loss_nz, p90_loss_nz))
    print("==================================================")

    print("\nSaving predictions and targets to CSV files...")

    # save
    output_dir = os.path.join(model_folder, "predictions")
    os.makedirs(output_dir, exist_ok=True)

    p10_forecast.to_csv(os.path.join(output_dir, "predictions_p10.csv"), index=False)
    p50_forecast.to_csv(os.path.join(output_dir, "predictions_p50.csv"), index=False)
    p90_forecast.to_csv(os.path.join(output_dir, "predictions_p90.csv"), index=False)
    targets.to_csv(os.path.join(output_dir, "targets.csv"), index=False)

    print(f"✅ Saved individual prediction CSVs to {output_dir}")

    merged = p50_forecast.copy()
    for col in [c for c in targets.columns if c.startswith("t+")]:
        merged[f"{col}_true"] = targets[col]
    merged_path = os.path.join(output_dir, "p50_vs_target.csv")
    merged.to_csv(merged_path, index=False)
    print(f"✅ Saved merged comparison CSV to {merged_path}")

    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_feature_importance(importance: dict, model, save_dir: str):
        import os, numpy as np, pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from data_formatters.base import InputTypes

        os.makedirs(save_dir, exist_ok=True)

        col_defs = model.column_definition
        input_feature_names = [name for (name, _, itype) in col_defs
                            if itype not in {InputTypes.ID, InputTypes.TIME}]

        num_cat = len(model.category_counts)
        num_reg = model.input_size - num_cat

        def names_for_static():
            names = []
            for i in range(num_reg):
                if i in model._static_input_loc:
                    names.append(input_feature_names[i])
            for j in range(num_cat):
                g = j + num_reg
                if g in model._static_input_loc:
                    names.append(input_feature_names[g])
            return names

        def known_global_indices():
            known_reg = [i for i in model._known_regular_input_idx
                        if i not in model._static_input_loc]
            known_cat = []
            for j in model._known_categorical_input_idx:
                g = j + num_reg
                if g not in model._static_input_loc:
                    known_cat.append(g)
            return known_reg + known_cat

        def unknown_global_indices():
            obs = set(model._input_obs_loc)
            known_reg_set = set(model._known_regular_input_idx)
            known_cat_local_set = set(model._known_categorical_input_idx)

            unknown_reg = [i for i in range(num_reg)
                        if (i not in known_reg_set)
                        and (i not in obs)
                        and (i not in model._static_input_loc)]
            unknown_cat = []
            for j in range(num_cat):
                g = j + num_reg
                if (j not in known_cat_local_set) and (g not in obs):
                    unknown_cat.append(g)
            return unknown_reg + unknown_cat

        def obs_global_indices():
            return list(model._input_obs_loc)

        def names_for_historical():
            idxs = unknown_global_indices() + known_global_indices() + obs_global_indices()
            return [input_feature_names[i] for i in idxs]

        def names_for_future():
            idxs = known_global_indices()
            return [input_feature_names[i] for i in idxs]

        # three types importance
        def get_feature_names(kind: str):
            if kind == "static_importance":
                return names_for_static()
            elif kind == "historical_importance":
                return names_for_historical()
            elif kind == "future_importance":
                return names_for_future()
            else:
                L = len(next(iter(importance.values())))
                return [f"f{i}" for i in range(L)]

        for name, values in importance.items():
            feature_names = get_feature_names(name)
            values = np.asarray(values).reshape(-1)
            if len(feature_names) != len(values):
                raise ValueError(
                    f"[{name}] nums ({len(feature_names)}) and values({len(values)}) mismatch. "
                    f"Check feature extraction logic."
                )

            df_imp = pd.DataFrame({"feature": feature_names, "importance": values})
            csv_path = os.path.join(save_dir, f"{name}.csv")
            df_imp.to_csv(csv_path, index=False)
            print(f"✅ feature importance saved to: {csv_path}")

            plt.figure(figsize=(max(8, len(feature_names) * 0.4), 4))
            sns.barplot(x="feature", y="importance", data=df_imp)
            plt.title(name.replace("_", " ").title())
            plt.xlabel("Feature Name")
            plt.ylabel("Mean Weight")
            plt.xticks(rotation=60, ha="right")
            plt.tight_layout()
            img_path = os.path.join(save_dir, f"{name}.png")
            plt.savefig(img_path, dpi=300)
            plt.close()
            print(f"🎨 feature importance saved to: {img_path}")

    def plot_attention_heatmap(attn_matrix: np.ndarray, num_encoder_steps: int, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        npy_path = os.path.join(save_dir, "attention_matrix.npy")
        csv_path = os.path.join(save_dir, "attention_matrix.csv")
        np.save(npy_path, attn_matrix)
        pd.DataFrame(attn_matrix).to_csv(csv_path, index=False)
        print(f"✅ attention matrix: {npy_path} and {csv_path}")

        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_matrix, cmap="viridis", cbar_kws={"label": "Attention Weight"})
        plt.title("Average Decoder Self-Attention")
        plt.axvline(x=num_encoder_steps, color="red", linestyle="--", linewidth=2)
        plt.tight_layout()


    save_dir = os.path.join(model_folder, "analysis_plots")
    os.makedirs(save_dir, exist_ok=True)

    # feature importance
    importance = best_model.get_feature_importance(valid, batch_size=64)
    plot_feature_importance(importance, best_model, save_dir)

    # attention
    avg_attn = best_model.get_attention(valid, batch_size=64)
    plot_attention_heatmap(avg_attn, best_model.num_encoder_steps, save_dir)


if __name__ == "__main__":

    def get_args():
        experiment_names = ExperimentConfig.default_experiments
        parser = argparse.ArgumentParser(description="PyTorch TFT training")
        parser.add_argument("expt_name", metavar="e", type=str, nargs="?",
                            default="volatility", choices=experiment_names)
        parser.add_argument("output_folder", metavar="f", type=str, nargs="?",
                            default=".", help="output folder")
        parser.add_argument("use_gpu", metavar="g", type=str, nargs="?",
                            choices=["yes", "no"], default="no")
        args = parser.parse_known_args()[0]
        root_folder = None if args.output_folder == "." else args.output_folder
        return args.expt_name, root_folder, args.use_gpu == "yes"

    name, output_folder, use_gpu_flag = get_args()
    print("Using output folder {}".format(output_folder))

    config = ExperimentConfig(name, output_folder)
    formatter = config.make_data_formatter()

    main(
        expt_name=name,
        use_gpu=use_gpu_flag,
        model_folder=os.path.join(config.model_folder, "fixed"),
        data_csv_path=config.data_csv_path,
        data_formatter=formatter,
        use_testing_mode=False,
    )
