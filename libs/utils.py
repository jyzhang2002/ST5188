import os
import pathlib
from typing import List, Sequence, Set, Any, Dict, Optional

import numpy as np
import torch


# ----------------------------------------------------------------------
# column helpers
# ----------------------------------------------------------------------
def get_single_col_by_input_type(input_type: Any, column_definition: Sequence[Sequence[Any]]) -> str:
    """Returns name of single column.

    Args:
        input_type: Input type of column to extract
        column_definition: Column definition list for experiment
    """
    cols = [tup[0] for tup in column_definition if tup[2] == input_type]
    if len(cols) != 1:
        raise ValueError(f"Invalid number of columns for {input_type}: {cols}")
    return cols[0]


def extract_cols_from_data_type(
    data_type: Any,
    column_definition: Sequence[Sequence[Any]],
    excluded_input_types: Set[Any],
) -> List[str]:
    """Extracts the names of columns that correspond to a defined data_type."""
    return [
        tup[0]
        for tup in column_definition
        if tup[1] == data_type and tup[2] not in excluded_input_types
    ]


# ----------------------------------------------------------------------
# loss functions
# ----------------------------------------------------------------------
def torch_quantile_loss(y_pred: torch.Tensor, y_true: torch.Tensor, quantile: float) -> torch.Tensor:
    """Standard quantile loss, same formula as TFT paper.

    Args:
        y_pred: (..., output_size)
        y_true: (..., output_size)
        quantile: float in (0,1)
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"Illegal quantile value={quantile}! Must be in (0,1).")

    diff = y_true - y_pred
    # max(q*diff, (q-1)*diff)
    loss = torch.maximum(quantile * diff, (quantile - 1) * diff)
    return loss.mean(dim=-1)


def numpy_normalised_quantile_loss(y: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """q-Risk metric in TFT.

    Args:
        y: (N, D) or (N,)
        y_pred: same shape
    """
    prediction_underflow = y - y_pred
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.0) + (
        1.0 - quantile
    ) * np.maximum(-prediction_underflow, 0.0)
    quantile_loss = weighted_errors.mean()
    normaliser = np.abs(y).mean()
    return 2 * quantile_loss / normaliser


# ----------------------------------------------------------------------
# filesystem
# ----------------------------------------------------------------------
def create_folder_if_not_exist(directory: str) -> None:
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# torch save / load
# ----------------------------------------------------------------------
def save_torch(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    model_folder: str,
    cp_name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Saves PyTorch model to checkpoint (.pt)."""
    create_folder_if_not_exist(model_folder)
    save_path = os.path.join(model_folder, f"{cp_name}.pt")
    payload = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, save_path)
    print(f"[utils] Model saved to: {save_path}")
    return save_path


def load_torch(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    model_folder: str,
    cp_name: str,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Loads PyTorch model from checkpoint (.pt)."""
    load_path = os.path.join(model_folder, f"{cp_name}.pt")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"[utils] checkpoint not found: {load_path}")

    payload = torch.load(load_path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    print(f"[utils] Model loaded from: {load_path}")
    return payload.get("extra", {})
