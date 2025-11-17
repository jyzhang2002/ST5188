import argparse
import datetime as dte
import os
import numpy as np
import pandas as pd
import torch

import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import tft_low.libs.tft_model_original
import libs.utils as utils

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model_original.TemporalFusionTransformer


def _to_python_scalars(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.generic,)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def main(expt_name, use_gpu, restart_opt, model_folder,
         hyperparam_iterations, data_csv_path, data_formatter):

    if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
        raise ValueError("Data formatter must inherit from GenericDataFormatter")

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    print(f"### Running hyperparameter optimization for {expt_name} on {device} ###")

    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train, valid, test = data_formatter.split_data(raw_data)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

    # fixed (dataset-defined) stuff
    fixed_params = data_formatter.get_experiment_params()
    if "column_definition" not in fixed_params or not isinstance(
        fixed_params["column_definition"], (list, tuple)
    ):
        fixed_params["column_definition"] = data_formatter.get_column_definition()
    fixed_params["model_folder"] = model_folder

    # searchable stuff
    param_ranges = ModelClass.get_hyperparm_choices()

    print("*** Loading hyperparam manager ***")
    opt_manager = HyperparamOptManager(param_ranges, fixed_params, model_folder)

    success = opt_manager.load_results()
    if success and not restart_opt:
        print("Loaded results from previous run")
    else:
        print("Creating new hyperparameter optimisation run")
        opt_manager.clear()

    print("*** Running calibration ***")
    while len(opt_manager.results.columns) < hyperparam_iterations:
        ith = len(opt_manager.results.columns) + 1
        print(f"# Hyperparam iteration {ith}/{hyperparam_iterations}")

        params = opt_manager.get_next_parameters()
        params = _to_python_scalars(params)

        print("Training with params:")
        for k, v in params.items():
            print(f"  {k}: {v}")

        model = ModelClass(params, use_cudnn=use_gpu, device=device)

        if not model.training_data_cached():
            model.cache_batched_data(train, "train", num_samples=train_samples)
            model.cache_batched_data(valid, "valid", num_samples=valid_samples)

        model.fit()
        val_loss = model.evaluate(valid)

        if np.allclose(val_loss, 0.) or np.isnan(val_loss):
            print("Bad configuration (nan/0), skipping...")
            val_loss = np.inf

        opt_manager.update_score(params, val_loss, model)

    print("*** Running tests ***")
    best_params = opt_manager.get_best_params()
    best_params = _to_python_scalars(best_params)
    best_params["column_definition"] = data_formatter.get_column_definition()

    model = ModelClass(best_params, use_cudnn=use_gpu, device=device)
    model.load(opt_manager.hyperparam_folder)

    print("Computing best validation loss...")
    val_loss = model.evaluate(valid)

    print("Computing test loss...")
    output_map = model.predict(test, return_targets=True)
    targets = data_formatter.format_predictions(output_map["targets"])
    p50_forecast = data_formatter.format_predictions(output_map["p50"])
    p90_forecast = data_formatter.format_predictions(output_map["p90"])

    def extract(df):
        return df[[c for c in df.columns if c not in {"forecast_time", "identifier"}]]

    p50_loss = utils.numpy_normalised_quantile_loss(
        extract(targets), extract(p50_forecast), 0.5
    )
    p90_loss = utils.numpy_normalised_quantile_loss(
        extract(targets), extract(p90_forecast), 0.9
    )

    print("Hyperparam optimisation completed @ {}".format(dte.datetime.now()))
    print(f"Best validation loss = {val_loss}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"{k} = {v}")
    print("Normalised Quantile Loss for Test Data: "
          f"P50={p50_loss.mean():.6f}, P90={p90_loss.mean():.6f}")


if __name__ == "__main__":

    def get_args():
        exps = ExperimentConfig.default_experiments
        parser = argparse.ArgumentParser(description="PyTorch TFT hyperparam opt")
        parser.add_argument("expt_name", metavar="e", type=str, nargs="?",
                            default="volatility", choices=exps)
        parser.add_argument("output_folder", metavar="f", type=str, nargs="?",
                            default=".")
        parser.add_argument("use_gpu", metavar="g", type=str, nargs="?",
                            choices=["yes", "no"], default="no")
        parser.add_argument("restart_hyperparam_opt", metavar="o", type=str, nargs="?",
                            choices=["yes", "no"], default="yes")
        args = parser.parse_known_args()[0]
        root_folder = None if args.output_folder == "." else args.output_folder
        return args.expt_name, root_folder, args.use_gpu == "yes", (args.restart_hyperparam_opt == "yes")

    name, folder, use_gpu_flag, restart_flag = get_args()
    print(f"Using output folder {folder}")

    config = ExperimentConfig(name, folder)
    formatter = config.make_data_formatter()

    main(
        expt_name=name,
        use_gpu=use_gpu_flag,
        restart_opt=restart_flag,
        model_folder=os.path.join(config.model_folder, "main"),
        hyperparam_iterations=config.hyperparam_iterations,
        data_csv_path=config.data_csv_path,
        data_formatter=formatter,
    )
