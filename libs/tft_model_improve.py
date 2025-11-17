import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import pandas as pd
import json
import gc
import os
import shutil
from tqdm import trange
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Any
import data_formatters.base as base
InputTypes = base.InputTypes

import os
import gc
import json
import shutil
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import data_formatters.base as data_base
import libs.utils as utils 

InputTypes = data_base.InputTypes


# ------------------------
# 🔹 Utility Layers
# ------------------------
def linear_layer(size: int,
                 activation: Optional[str] = None,
                 use_time_distributed: bool = False,
                 use_bias: bool = True):
    """Simple linear layer builder."""
    layer = nn.Linear(in_features=None, out_features=size, bias=use_bias)
    return layer, activation


def apply_mlp(inputs: Tensor,
              hidden_size: int,
              output_size: int,
              output_activation: Optional[str] = None,
              hidden_activation: str = 'tanh') -> Tensor:
    """Applies simple feed-forward MLP."""
    hidden = F.__dict__[hidden_activation](inputs @ torch.randn(inputs.size(-1), hidden_size))
    out = hidden @ torch.randn(hidden_size, output_size)
    if output_activation is not None:
        out = F.__dict__[output_activation](out)
    return out


# ------------------------
# 🔹 Gated Linear Unit (GLU)
# ------------------------
class GatingLayer(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float = 0.0):
        super().__init__()
        # print("Initializing GatingLayer with hidden_size =", hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.dropout(x)
        act = self.fc(x)
        gate = torch.sigmoid(self.gate(x))
        return act * gate, gate

# ====================================================
# 🔹 Reversible Instance Normalization (RevIN)
# ====================================================
class RevINLayer(nn.Module):
    """
    Reversible Instance Normalization for real-valued features only.
    x: (B, T, F)
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta  = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self,
                x: torch.Tensor,
                mode: str,
                stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if mode == "normalize":
            # normalization
            mu  = x.mean(dim=1, keepdim=True)                       # (B,1,F)
            std = x.std(dim=1, keepdim=True, unbiased=False)        # (B,1,F)
            std = std + self.eps
            x_norm = self.gamma * (x - mu) / std + self.beta
            return x_norm, (mu, std)

        elif mode == "denormalize":
            assert stats is not None, "RevIN requires stats for denorm"
            mu, std = stats
            x_denorm = (x - self.beta) / (self.gamma + 1e-12)
            x_denorm = x_denorm * std + mu
            return x_denorm

        else:
            raise ValueError("mode must be 'normalize' or 'denormalize'")


# ------------------------
# 🔹 Gated Residual Network (GRN)
# ------------------------
class GatedResidualNetwork(nn.Module):
    def __init__(self, hidden_size: int, output_size: Optional[int] = None,
                 dropout_rate: float = 0.0, context_size: Optional[int] = None):
        super().__init__()
        if output_size is None:
            output_size = hidden_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.context_layer = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.gate = GatingLayer(output_size, dropout_rate)
        self.skip = nn.Linear(hidden_size, output_size) if output_size != hidden_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        residual = self.skip(x)
        out = self.linear1(x)
        if context is not None:
            out = out + self.context_layer(context)
        out = F.elu(out)
        out = self.linear2(out)
        gated, _ = self.gate(out)
        return self.layer_norm(residual + gated)


# ------------------------
# 🔹 Scaled Dot-Product Attention
# ------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


# ------------------------
# 🔹 Interpretable Multi-Head Attention
# ------------------------
class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = self.d_v = d_model // n_head
        self.qs_layers = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)])
        self.ks_layers = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)])
        shared_vs = nn.Linear(d_model, self.d_v, bias=False)
        self.vs_layers = nn.ModuleList([shared_vs for _ in range(n_head)])
        self.attention = ScaledDotProductAttention(dropout)
        # self.fc = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        head_outputs, attns = [], []
        for i in range(self.n_head):
            q_i, k_i, v_i = self.qs_layers[i](q), self.ks_layers[i](k), self.vs_layers[i](v)
            out, attn = self.attention(q_i, k_i, v_i, mask)
            head_outputs.append(self.dropout(out))
            attns.append(attn)
        stacked_heads = torch.stack(head_outputs)
        stacked_attns = torch.stack(attns)
        output = stacked_heads.mean(dim=0)
        output = self.fc(output)
        output = self.dropout(output)
        return output, stacked_attns


# ------------------------
# 🔹 Decoder Mask (causal mask)
# ------------------------
def get_decoder_mask(x: Tensor) -> Tensor:
    """Creates a causal mask."""
    seq_len = x.size(1)
    mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0)
    return mask


# ------------------------
# 🔹 TFT Data Cache (for batching and reuse)
# ------------------------
class TFTDataCache:
    """Caches data (numpy/pandas) for the TFT."""
    _cache: Dict[str, Any] = {}

    @classmethod
    def update(cls, data: Any, key: str):
        cls._cache[key] = data

    @classmethod
    def get(cls, key: str) -> Any:
        if key not in cls._cache:
            raise KeyError(f"No cached data for key '{key}'")
        data = cls._cache[key]
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, np.ndarray):
            return data.copy()
        else:
            return data

    @classmethod
    def contains(cls, key: str) -> bool:
        return key in cls._cache


# ----------------------------------------------------------------------
# functions
# ----------------------------------------------------------------------
def torch_quantile_loss(y_pred: torch.Tensor,
                        y_true: torch.Tensor,
                        quantiles: List[float]) -> torch.Tensor:
    """
    y_pred: (B, T, out_size * num_q)
    y_true: (B, T, out_size)
    """
    assert y_pred.dim() == 3 and y_true.dim() == 3
    B, T, out = y_true.size()
    num_q = len(quantiles)
    losses = []
    for qi, q in enumerate(quantiles):
        pred_q = y_pred[..., qi * out:(qi + 1) * out]
        diff = y_true - pred_q
        loss_q = torch.maximum(q * diff, (q - 1) * diff).mean()
        losses.append(loss_q)
    return sum(losses)


# ----------------------------------------------------------------------
# TemporalVariableSelectionNetwork
# ----------------------------------------------------------------------
class TemporalVariableSelectionNetwork(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_inputs: int,
                 hidden_size: int,
                 dropout: float,
                 context_size: Optional[int] = None):
        super().__init__()
        self.num_inputs = num_inputs
        self.d_model = d_model
        self.flatten = nn.Linear(d_model * num_inputs, hidden_size)
        self.var_selection_grn = GatedResidualNetwork(
            hidden_size,
            output_size=num_inputs,
            dropout_rate=dropout,
            context_size=context_size
        )
        self.softmax = nn.Softmax(dim=-1)
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                d_model,
                output_size=hidden_size,
                dropout_rate=dropout,
                context_size=None
            ) for _ in range(num_inputs)
        ])

    def forward(self, embedding: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # embedding: (B, T, d_model, num_inputs)
        B, T, D, N = embedding.size()
        flat = embedding.reshape(B, T, D * N)  # (B, T, D*N)
        x = self.flatten(flat)  # (B, T, hidden)
        if context is not None:
            # context: (B, hidden) -> (B, T, hidden)
            context = context.unsqueeze(1).expand(-1, T, -1)
        weights = self.var_selection_grn(x, context)  # (B, T, num_inputs)
        weights = self.softmax(weights)  # softmax over variables
        v_list = []
        for i in range(N):
            v_i = embedding[..., i]  # (B, T, D)
            v_i = self.variable_grns[i](v_i)  # (B, T, hidden)
            v_list.append(v_i)
        v_stack = torch.stack(v_list, dim=-1)  # (B, T, hidden, N)
        weights = weights.unsqueeze(2)  # (B, T, 1, N)
        combined = (v_stack * weights).sum(-1)  # (B, T, hidden)
        return combined, weights.squeeze(2)  # weights: (B, T, N)


# ----------------------------------------------------------------------
# StaticVariableSelectionNetwork
# ----------------------------------------------------------------------
class StaticVariableSelectionNetwork(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_static: int,
                 dropout: float):
        super().__init__()
        self.num_static = num_static
        self.flatten = nn.Linear(num_static * d_model, d_model)
        self.mlp = GatedResidualNetwork(
            d_model,
            output_size=num_static,
            dropout_rate=dropout,
            context_size=None
        )
        self.softmax = nn.Softmax(dim=-1)
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(
                d_model,
                output_size=d_model,
                dropout_rate=dropout,
                context_size=None
            ) for _ in range(num_static)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, S, D)
        B, S, D = x.size()
        flat = x.reshape(B, S * D)
        hidden = self.flatten(flat)
        logits = self.mlp(hidden)  # (B, S)
        weights = self.softmax(logits).unsqueeze(-1)  # (B, S, 1)
        trans = []
        for i in range(S):
            trans.append(self.var_grns[i](x[:, i, :]))  # (B, D)
        trans = torch.stack(trans, dim=1)  # (B, S, D)
        combined = (weights * trans).sum(dim=1)  # (B, D)
        return combined, weights.squeeze(-1)  # (B, D), (B, S)


# ----------------------------------------------------------------------
# # TemporalFusionTransformer
# ----------------------------------------------------------------------
class TemporalFusionTransformer(nn.Module):
    """
    PyTorch implementation of TFT.
    """

    def __init__(self, raw_params: Dict[str, Any], use_cudnn: bool = True, device: Optional[str] = None):
        super().__init__()

        self.name = self.__class__.__name__
        params = dict(raw_params)

        self.time_steps: int = int(params["total_time_steps"])
        self.input_size: int = int(params["input_size"])
        self.output_size: int = int(params["output_size"])
        self.category_counts: List[int] = json.loads(str(params["category_counts"]))
        self.n_multiprocessing_workers: int = int(params["multiprocessing_workers"])

        self._input_obs_loc: List[int] = json.loads(str(params["input_obs_loc"]))
        self._static_input_loc: List[int] = json.loads(str(params["static_input_loc"]))
        self._known_regular_input_idx: List[int] = json.loads(str(params["known_regular_inputs"]))
        self._known_categorical_input_idx: List[int] = json.loads(str(params["known_categorical_inputs"]))

        self.column_definition = params["column_definition"]

        # ===== parameters =====
        self.quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.use_cudnn = use_cudnn
        self.hidden_layer_size: int = int(params["hidden_layer_size"])
        self.dropout_rate: float = float(params["dropout_rate"])
        self.max_gradient_norm: float = float(params["max_gradient_norm"])
        self.learning_rate: float = float(params["learning_rate"])
        self.minibatch_size: int = int(params["minibatch_size"])
        self.num_epochs: int = int(params["num_epochs"])
        self.early_stopping_patience: int = int(params["early_stopping_patience"])

        self.num_encoder_steps: int = int(params["num_encoder_steps"])
        self.num_stacks: int = int(params["stack_size"])
        self.num_heads: int = int(params["num_heads"])

        self._temp_folder = os.path.join(params["model_folder"], "tmp")
        self.reset_temp_folder()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        print(f"*** {self.name} params ***")
        for k in params:
            print(f"# {k} = {params[k]}")

        self._build_network()

        # device
        self.to(self.device)

        # attention components
        self._attention_components = None
        self._prediction_parts = None

    # ------------------------------------------------------------------
    # build network
    # ------------------------------------------------------------------
    def _build_network(self):
        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables
        self.num_categorical_variables = num_categorical_variables
        self.num_regular_variables = num_regular_variables

        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables
        self.num_categorical_variables = num_categorical_variables
        self.num_regular_variables = num_regular_variables

        self.revin_cont = RevINLayer(num_features=self.num_regular_variables)

        # 1) embedding
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cnt,
                         embedding_dim=self.hidden_layer_size)
            for cnt in self.category_counts
        ])

        # 2) embedding (time-distributed dense)
        self.regular_variable_projection = nn.Linear(1, self.hidden_layer_size)

        # 3) StaticVariableSelectionNetwork
        self.has_static = len(self._static_input_loc) > 0
        if self.has_static:
            num_static = len(self._static_input_loc)
            self.static_vsn = StaticVariableSelectionNetwork(
                d_model=self.hidden_layer_size,
                num_static=num_static,
                dropout=self.dropout_rate
            )
            # static context
            self.static_context_varsel = GatedResidualNetwork(
                self.hidden_layer_size, dropout_rate=self.dropout_rate
            )
            self.static_context_enrichment = GatedResidualNetwork(
                self.hidden_layer_size, dropout_rate=self.dropout_rate
            )
            self.static_context_state_h = GatedResidualNetwork(
                self.hidden_layer_size, dropout_rate=self.dropout_rate
            )
            self.static_context_state_c = GatedResidualNetwork(
                self.hidden_layer_size, dropout_rate=self.dropout_rate
            )

        # 4) vsn
        self.temporal_vsn_hist = None
        self.temporal_vsn_fut = None

        # 5) LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.hidden_layer_size,
            hidden_size=self.hidden_layer_size,
            num_layers=1,
            batch_first=True
        )
        self.decoder_lstm = nn.LSTM(
            input_size=self.hidden_layer_size,
            hidden_size=self.hidden_layer_size,
            num_layers=1,
            batch_first=True
        )

        # 6) self-attention
        self.self_attn = InterpretableMultiHeadAttention(
            n_head=self.num_heads,
            d_model=self.hidden_layer_size,
            dropout=self.dropout_rate
        )

        # 7) decoder
        self.post_attn_gating = GatingLayer(self.hidden_layer_size, dropout_rate=self.dropout_rate)
        self.post_attn_layernorm = nn.LayerNorm(self.hidden_layer_size)

        self.decoder_grn = GatedResidualNetwork(
            self.hidden_layer_size, dropout_rate=self.dropout_rate
        )
        self.decoder_gating = GatingLayer(self.hidden_layer_size)
        self.decoder_layernorm = nn.LayerNorm(self.hidden_layer_size)

        # 8) output
        self.output_projection = nn.Linear(
            self.hidden_layer_size,
            self.output_size * len(self.quantiles)
        )

        self._optimizer = None

    # ------------------------------------------------------------------
    # embed inputs
    # ------------------------------------------------------------------
    def _embed_inputs(self, all_inputs: torch.Tensor):
        B, T, F = all_inputs.size()
        num_cat = self.num_categorical_variables
        num_reg = self.num_regular_variables

        regular_inputs = all_inputs[:, :, :num_reg]  # (B, T, num_reg)
        cat_inputs = all_inputs[:, :, num_reg:]      # (B, T, num_cat)

        B, T, num_reg = regular_inputs.shape
        reg_emb = torch.empty(B, T, self.hidden_layer_size, num_reg, device=regular_inputs.device)
        for i in range(num_reg):
            x = regular_inputs[:, :, i:i+1]
            reg_emb[..., i] = self.regular_variable_projection(x).squeeze(-1)

        # embedding
        cat_emb_list = []
        for i in range(num_cat):
            idx = cat_inputs[:, :, i].long()   # (B, T)
            e = self.categorical_embeddings[i](idx)  # (B, T, d_model)
            cat_emb_list.append(e)
        if num_cat > 0:
            cat_emb = torch.stack(cat_emb_list, dim=-1)  # (B, T, d_model, num_cat)
        else:
            cat_emb = None

        # static inputs
        if self.has_static:
            static_emb_list = []
            # static
            for i in range(num_reg):
                if i in self._static_input_loc:
                    v = self.regular_variable_projection(regular_inputs[:, 0:1, i:i+1])  # (B,1,d_model)
                    static_emb_list.append(v[:, 0, :])
            # static
            for i in range(num_cat):
                if i + num_reg in self._static_input_loc:
                    v = self.categorical_embeddings[i](cat_inputs[:, 0, i].long())  # (B, d_model)
                    static_emb_list.append(v)
            static_inputs = torch.stack(static_emb_list, dim=1)  # (B, S, d_model)
        else:
            static_inputs = None

        obs_emb_list = []
        for i in self._input_obs_loc:
            if i < num_reg:
                obs_emb_list.append(reg_emb[..., i])
            else:
                obs_emb_list.append(cat_emb[..., i - num_reg])
        obs_emb = torch.stack(obs_emb_list, dim=-1)  # (B, T, d_model, obs_num)

        known_emb_list = []
        for i in self._known_regular_input_idx:
            if i not in self._static_input_loc:
                known_emb_list.append(reg_emb[..., i])
        for i in self._known_categorical_input_idx:
            if i + num_reg not in self._static_input_loc:
                known_emb_list.append(cat_emb[..., i])
        if known_emb_list:
            known_emb = torch.stack(known_emb_list, dim=-1)  # (B, T, d_model, K)
        else:
            known_emb = None

        # unknown inputs
        unknown_emb_list = []
        for i in range(num_reg):
            if (i not in self._known_regular_input_idx) and (i not in self._input_obs_loc):
                unknown_emb_list.append(reg_emb[..., i])
        for i in range(num_cat):
            gidx = i + num_reg
            if (i not in self._known_categorical_input_idx) and (gidx not in self._input_obs_loc):
                unknown_emb_list.append(cat_emb[..., i])
        if unknown_emb_list:
            unknown_emb = torch.stack(unknown_emb_list, dim=-1)  # (B, T, d_model, U)
        else:
            unknown_emb = None

        return unknown_emb, known_emb, obs_emb, static_inputs

    # ------------------------------------------------------------------
    # add RevIN
    # ------------------------------------------------------------------
    def forward(self, all_inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = all_inputs.size()
        device = all_inputs.device

        # ===== 1) RevIN =====
        num_reg = self.num_regular_variables
        num_cat = self.num_categorical_variables
        assert num_reg + num_cat == D, f"Input size mismatch: {D} vs {num_reg}+{num_cat}"

        reg_inputs = all_inputs[:, :, :num_reg]       # (B,T,num_reg)
        cat_inputs = all_inputs[:, :, num_reg:]       # (B,T,num_cat)

        reg_norm, _ = self.revin_cont(reg_inputs, mode="normalize")
        all_inputs_norm = torch.cat([reg_norm, cat_inputs], dim=-1) 

        unknown_emb, known_emb, obs_emb, static_inputs = self._embed_inputs(all_inputs_norm)

        enc_len = self.num_encoder_steps
        dec_len = self.time_steps - self.num_encoder_steps

        # 2) static vsn
        if static_inputs is not None:
            static_encoder, static_flags = self.static_vsn(static_inputs)
            static_ctx_varsel = self.static_context_varsel(static_encoder)
            static_ctx_enrich = self.static_context_enrichment(static_encoder)
            static_state_h = self.static_context_state_h(static_encoder)
            static_state_c = self.static_context_state_c(static_encoder)
        else:
            static_flags = None
            static_ctx_varsel = None
            static_ctx_enrich = None
            static_state_h = torch.zeros(B, self.hidden_layer_size, device=device)
            static_state_c = torch.zeros(B, self.hidden_layer_size, device=device)

        # 3) hist
        hist_list = []
        if unknown_emb is not None:
            hist_list.append(unknown_emb[:, :enc_len, :, :])
        if known_emb is not None:
            hist_list.append(known_emb[:, :enc_len, :, :])
        hist_list.append(obs_emb[:, :enc_len, :, :])
        hist_cat = torch.cat(hist_list, dim=-1)
        n_hist_vars = hist_cat.size(-1)

        # 4) future
        fut_list = []
        if known_emb is not None:
            fut_list.append(known_emb[:, enc_len:, :, :])
        fut_cat = torch.cat(fut_list, dim=-1) if fut_list else None
        n_fut_vars = fut_cat.size(-1) if fut_cat is not None else 0

        if self.temporal_vsn_hist is None:
            self.temporal_vsn_hist = TemporalVariableSelectionNetwork(
                d_model=self.hidden_layer_size,
                num_inputs=n_hist_vars,
                hidden_size=self.hidden_layer_size,
                dropout=self.dropout_rate,
                context_size=self.hidden_layer_size if static_ctx_varsel is not None else None
            ).to(device)

        if self.temporal_vsn_fut is None and n_fut_vars > 0:
            self.temporal_vsn_fut = TemporalVariableSelectionNetwork(
                d_model=self.hidden_layer_size,
                num_inputs=n_fut_vars,
                hidden_size=self.hidden_layer_size,
                dropout=self.dropout_rate,
                context_size=self.hidden_layer_size if static_ctx_varsel is not None else None
            ).to(device)

        hist_ctx, hist_flags = self.temporal_vsn_hist(hist_cat, static_ctx_varsel)
        if fut_cat is not None:
            fut_ctx, fut_flags = self.temporal_vsn_fut(fut_cat, static_ctx_varsel)
        else:
            fut_ctx = torch.zeros(B, dec_len, self.hidden_layer_size, device=device)
            fut_flags = torch.zeros(B, dec_len, 0, device=device)

        h0 = static_state_h.unsqueeze(0)
        c0 = static_state_c.unsqueeze(0)
        enc_out, (enc_h, enc_c) = self.encoder_lstm(hist_ctx, (h0, c0))
        dec_out, _ = self.decoder_lstm(fut_ctx, (enc_h, enc_c))

        lstm_out = torch.cat([enc_out, dec_out], dim=1)
        input_embed_full = torch.cat([hist_ctx, fut_ctx], dim=1)

        # gated skip connection
        glu = GatingLayer(self.hidden_layer_size, dropout_rate=self.dropout_rate).to(device)
        lstm_gated, _ = glu(lstm_out)
        temporal_feature_layer = F.layer_norm(lstm_gated + input_embed_full,
                                              normalized_shape=[self.hidden_layer_size])

        # static enrichment
        if static_ctx_enrich is not None:
            expanded_static = static_ctx_enrich.unsqueeze(1).expand(-1, T, -1)
            enriched = GatedResidualNetwork(
                self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                context_size=self.hidden_layer_size
            ).to(device)(temporal_feature_layer, expanded_static)
        else:
            enriched = temporal_feature_layer

        # decoder self-attention
        mask = get_decoder_mask(enriched)
        x, self_att = self.self_attn(enriched, enriched, enriched, mask=mask)

        x_gated, _ = self.post_attn_gating(x)
        x = self.post_attn_layernorm(x_gated + enriched)
        x = self.decoder_grn(x)
        x2, _ = self.decoder_gating(x)
        transformer_out = self.decoder_layernorm(x2 + x)

        # decoder
        decoder_out = transformer_out[:, self.num_encoder_steps:, :]
        out = self.output_projection(decoder_out)  # (B, dec_len, out_size * num_q)

        attention_components = {
            "decoder_self_attn": self_att,
            "static_flags": static_flags,
            "historical_flags": hist_flags,
            "future_flags": fut_flags
        }

        return out, attention_components

    # ------------------------------------------------------------------
    # cache data functions
    # ------------------------------------------------------------------
    def training_data_cached(self) -> bool:
        return TFTDataCache.contains("train") and TFTDataCache.contains("valid")

    def cache_batched_data(self, data: pd.DataFrame, cache_key: str, num_samples: int = -1):
        if num_samples > 0:
            cached = self._batch_sampled_data(data, max_samples=num_samples)
        else:
            cached = self._batch_data(data)
        TFTDataCache.update(cached, cache_key)
        print(f'Cached data "{cache_key}" updated')

    # ------------------------------------------------------------------
    # batch data functions
    # ------------------------------------------------------------------
    def _get_single_col_by_type(self, input_type):
        return utils.get_single_col_by_input_type(input_type, self.column_definition)

    def _batch_sampled_data(self, data: pd.DataFrame, max_samples: int):
        if max_samples < 1:
            raise ValueError(f"Illegal number of samples: {max_samples}")

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        data = data.sort_values(by=[id_col, time_col])

        valid_locs = []
        split_map = {}
        for identifier, df in data.groupby(id_col):
            num_entries = len(df)
            if num_entries >= self.time_steps:
                valid_locs += [
                    (identifier, self.time_steps + i)
                    for i in range(num_entries - self.time_steps + 1)
                ]
            split_map[identifier] = df
        
        print(f"✅ Total valid samples available: {len(valid_locs)}")
        print(f"✅ Selected samples: {max_samples}")

        inputs = np.zeros((max_samples, self.time_steps, self.input_size), dtype=np.float32)
        outputs = np.zeros((max_samples, self.time_steps, self.output_size), dtype=np.float32)
        time = np.empty((max_samples, self.time_steps, 1), dtype=object)
        identifiers = np.empty((max_samples, self.time_steps, 1), dtype=object)

        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0]
            for tup in self.column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        if len(valid_locs) > max_samples:
            chosen = np.random.choice(len(valid_locs), max_samples, replace=False)
            ranges = [valid_locs[i] for i in chosen]
        else:
            ranges = valid_locs

        for i, (identifier, end_idx) in enumerate(ranges):
            sliced = split_map[identifier].iloc[end_idx - self.time_steps:end_idx]
            inputs[i, :, :] = sliced[input_cols].values
            outputs[i, :, :] = sliced[[target_col]].values
            time[i, :, 0] = sliced[time_col].values
            identifiers[i, :, 0] = sliced[id_col].values

        data_map = {
            "inputs": inputs,
            "outputs": outputs[:, self.num_encoder_steps:, :],
            "active_entries": np.ones_like(outputs[:, self.num_encoder_steps:, :]),
            "time": time,
            "identifier": identifiers,
        }
        return data_map

    def _batch_data(self, data: pd.DataFrame):
        def _batch_single_entity(df_slice: pd.DataFrame):
            time_steps = len(df_slice)
            lags = self.time_steps
            x = df_slice.values
            if time_steps >= lags:
                return np.stack(
                    [x[i:time_steps - (lags - 1) + i, :] for i in range(lags)],
                    axis=1,
                )
            else:
                return None

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0] for tup in self.column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        data_map: Dict[str, List[np.ndarray]] = {}
        for _, sliced in data.groupby(id_col):
            col_mappings = {
                "identifier": [id_col],
                "time": [time_col],
                "outputs": [target_col],
                "inputs": input_cols,
            }
            for k, cols in col_mappings.items():
                arr = _batch_single_entity(sliced[cols].copy())
                if arr is None:
                    continue
                if k not in data_map:
                    data_map[k] = [arr]
                else:
                    data_map[k].append(arr)

        # concat
        for k in data_map:
            data_map[k] = np.concatenate(data_map[k], axis=0)

        data_map["outputs"] = data_map["outputs"][:, self.num_encoder_steps:, :]
        active_entries = np.ones_like(data_map["outputs"])
        data_map["active_entries"] = active_entries

        print(f"windows nums: {len(data_map['inputs'])}")
        print(f"input data shape: {data_map['inputs'].shape}") 
        print(f"output data shape: {data_map['outputs'].shape}")

        return data_map

    # ------------------------------------------------------------------
    # train / eval functions
    # ------------------------------------------------------------------
    def _to_tensor_batch(self, batch_map: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(batch_map["inputs"]).float().to(self.device)
        y = torch.from_numpy(batch_map["outputs"]).float().to(self.device)
        mask = torch.from_numpy((np.sum(batch_map["active_entries"], axis=-1) > 0.0).astype(np.float32)).to(self.device)
        return x, y, mask

    def fit(self, train_df: Optional[pd.DataFrame] = None,
            valid_df: Optional[pd.DataFrame] = None):
        print(f"*** Fitting {self.name} (PyTorch) ***")

        if train_df is None:
            train_data = TFTDataCache.get("train")
        else:
            train_data = self._batch_data(train_df)

        if valid_df is None:
            valid_data = TFTDataCache.get("valid")
        else:
            valid_data = self._batch_data(valid_df)

        x_train, y_train, m_train = self._to_tensor_batch(train_data)
        x_val, y_val, m_val = self._to_tensor_batch(valid_data)

        dataset_size = x_train.size(0)
        val_size = x_val.size(0)

        self._optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate
        )

        best_val = float("inf")
        no_improve = 0

        for epoch in trange(self.num_epochs):
            self.train()
            perm = torch.randperm(dataset_size)
            x_train = x_train[perm]
            y_train = y_train[perm]
            m_train = m_train[perm]

            epoch_loss = 0.0
            num_batches = int(np.ceil(dataset_size / self.minibatch_size))
            for b in trange(num_batches):
                start = b * self.minibatch_size
                end = min((b + 1) * self.minibatch_size, dataset_size)
                xb = x_train[start:end]
                yb = y_train[start:end]
                mb = m_train[start:end]

                self._optimizer.zero_grad()
                preds, _ = self(xb)
                loss = torch_quantile_loss(preds, yb, self.quantiles)
                # mask
                loss = (loss * 1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_gradient_norm)
                self._optimizer.step()
                epoch_loss += loss.item()

            self.eval()
            with torch.no_grad():
                batch_size = 32
                preds_val_list = []
                
                for i in trange(0, len(x_val), batch_size):
                    end_idx = min(i + batch_size, len(x_val))
                    x_batch = x_val[i:end_idx]
                    y_batch = y_val[i:end_idx]
                    
                    preds_batch, _ = self(x_batch)
                    preds_val_list.append(preds_batch)
                
                preds_val = torch.cat(preds_val_list, dim=0)
                val_loss = torch_quantile_loss(preds_val, y_val, self.quantiles).item()

            print(f"[Epoch {epoch+1}/{self.num_epochs}] train_loss={epoch_loss/num_batches:.6f} "
                  f"val_loss={val_loss:.6f}")

            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                no_improve = 0
                # save best
                self.save(self._temp_folder)
            else:
                no_improve += 1
                if no_improve >= self.early_stopping_patience:
                    print("Early stopping triggered.")
                    break
            
            torch.cuda.empty_cache()
            gc.collect()


        # load best model
        if os.path.exists(os.path.join(self._temp_folder, f"{self.name}.pt")):
            self.load(self._temp_folder)

    def evaluate(self, data=None, eval_metric='loss', batch_size: Optional[int] = None):
        """
        Evaluates model on validation/test data in mini-batches to avoid OOM.
        """
        self.eval()  # important for dropout/batchnorm
        torch.cuda.empty_cache()

        if data is None:
            print('Using cached validation data')
            raw_data = TFTDataCache.get('valid')
        else:
            raw_data = self._batch_data(data)

        inputs = torch.tensor(raw_data['inputs'], dtype=torch.float32)
        outputs = torch.tensor(raw_data['outputs'], dtype=torch.float32)
        # active_entries = torch.tensor(self._get_active_locations(raw_data['active_entries']), dtype=torch.float32)

        if batch_size is None:
            batch_size = getattr(self, 'minibatch_size', 64)

        n = inputs.shape[0]
        num_batches = int(np.ceil(n / batch_size))
        losses = []

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Evaluating", total=num_batches, leave=True):
                batch_slice = slice(i * batch_size, min((i + 1) * batch_size, n))
                xb = inputs[batch_slice].to(self.device)
                yb = outputs[batch_slice].to(self.device)

                preds, _ = self(xb)
                # preds shape: (B, T, output_size * len(quantiles))
                loss = torch_quantile_loss(preds, yb, self.quantiles)
                losses.append(loss.item())

                del xb, yb, preds
                torch.cuda.empty_cache()
                gc.collect()

        avg_loss = np.mean(losses)
        print(f'Validation loss (avg over {num_batches} batches): {avg_loss:.6f}')
        return avg_loss

    def predict(self, df: pd.DataFrame, return_targets: bool = False, batch_size: int = 512) -> Dict[str, pd.DataFrame]:
        """Memory-efficient prediction, supporting large datasets."""
        data_map = self._batch_data(df)
        x, y, _ = self._to_tensor_batch(data_map)
        time = data_map["time"]
        identifier = data_map["identifier"]

        self.eval()
        preds_list = []
        device = next(self.parameters()).device

        with torch.no_grad():
            # batch-wise prediction
            for start in trange(0, x.size(0), batch_size):
                end = min(start + batch_size, x.size(0))
                xb = x[start:end].to(device)
                preds_b, _ = self(xb)
                preds_list.append(preds_b.cpu())
            preds = torch.cat(preds_list, dim=0).numpy()

        B, dec_len, out_dim = preds.shape
        if self.output_size != 1:
            raise NotImplementedError("Only 1D target supported right now.")

        def format_outputs(arr_2d: np.ndarray, time_arr, id_arr) -> pd.DataFrame:
            df_out = pd.DataFrame(arr_2d, columns=[f"t+{i}" for i in range(dec_len)])
            df_out["forecast_time"] = time_arr[:, self.num_encoder_steps - 1, 0]
            df_out["identifier"] = id_arr[:, 0, 0]
            return df_out[["forecast_time", "identifier"] + [f"t+{i}" for i in range(dec_len)]]

        process_map = {}
        for qi, q in enumerate(self.quantiles):
            arr = preds[..., qi * self.output_size:(qi + 1) * self.output_size].squeeze(-1)
            process_map[f"p{int(q*100)}"] = format_outputs(arr, time, identifier)

        if return_targets:
            y_np = y.cpu().numpy().squeeze(-1)
            process_map["targets"] = format_outputs(y_np, time, identifier)
            self._calculate_point_metrics(process_map, y_np)

        return process_map


    def _calculate_point_metrics(self, process_map: Dict, targets: np.ndarray):
        # p50 predictions
        p50_key = None
        for key in process_map.keys():
            if key.startswith('p50') or ('p50' in key and 'p500' not in key):
                p50_key = key
                break
        
        if p50_key is None:
            print("Warning: Could not find P50 predictions for metric calculation")
            return
        
        # P50
        predictions_df = process_map[p50_key]
        pred_columns = [f"t+{i}" for i in range(targets.shape[1])]
        predictions = predictions_df[pred_columns].values
        
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        pred_flat = pred_flat[mask]
        target_flat = target_flat[mask]
        
        if len(pred_flat) == 0:
            print("Warning: No valid data for metric calculation")
            return
        
        # metric calculations
        mse = np.mean((pred_flat - target_flat) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_flat - target_flat))
        
        denominator = np.abs(pred_flat) + np.abs(target_flat)
        smape_values = 200 * np.abs(pred_flat - target_flat) / (np.abs(pred_flat) + np.abs(target_flat) + 1e-8)
        smape = np.mean(smape_values)
        
        print("\n" + "="*50)
        print("POINT FORECAST METRICS (using P50 as point forecast)")
        print("="*50)
        print(f"RMSE:  {rmse:.4f}")
        print(f"MAE:   {mae:.4f}")
        print(f"Sample size: {len(pred_flat)} predictions")
        print("="*50)
        
        metrics = {
            'rmse': rmse,
            'mae': mae, 
            'smape': smape,
            'sample_size': len(pred_flat)
        }
        return metrics

    # ------------------------------------------------------------------
    # attention
    # ------------------------------------------------------------------
    def get_attention(self, df: pd.DataFrame, batch_size: int = 64):
        """
        Returns *average* decoder self-attention across samples (memory-safe).
        Output: (T, T) numpy array
        """
        data_map = self._batch_data(df)
        x, _, _ = self._to_tensor_batch(data_map)

        self.eval()
        torch.cuda.empty_cache()

        n = x.size(0)
        attn_sum = None
        count = 0

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = x[start:end]

                _, att = self(xb)
                att_b = att["decoder_self_attn"]  # (num_heads, B, T, T)

                # mean over heads, then batch
                att_b = att_b.mean(dim=0).mean(dim=0)  # → (T, T)

                attn_sum = att_b if attn_sum is None else (attn_sum + att_b)
                count += 1

                del xb, att, att_b
                torch.cuda.empty_cache()

        avg_attn = (attn_sum / count).detach().cpu().numpy()
        print("✅ Average attention matrix computed (batched, memory-safe).")
        return avg_attn
    
    # ------------------------------------------------------------------
    # feature importance
    # ------------------------------------------------------------------
    def get_feature_importance(self, df: pd.DataFrame, batch_size: int = 64):
        """
        Memory-efficient computation of feature importance by batching.
        Output:
            {
            'static_importance': (num_static,),
            'historical_importance': (num_hist_features,),
            'future_importance': (num_fut_features,)
            }
        """
        data_map = self._batch_data(df)
        x, _, _ = self._to_tensor_batch(data_map)

        self.eval()
        torch.cuda.empty_cache()

        n = x.size(0)
        static_sum = None
        hist_sum = None
        fut_sum = None
        count = 0

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = x[start:end]

                _, att = self(xb)

                # ---- Static ----
                sf = att.get("static_flags", None)
                if sf is not None:
                    # (B, S) → sum over batch
                    sf_mean_b = sf.mean(dim=0)  # (S,)
                    static_sum = sf_mean_b if static_sum is None else static_sum + sf_mean_b

                # ---- Historical ----
                hf = att.get("historical_flags", None)
                if hf is not None and hf.numel() > 0:
                    # (B, enc_len, N_hist) → mean over batch & time
                    hf_mean_b = hf.mean(dim=0).mean(dim=0)  # (N_hist,)
                    hist_sum = hf_mean_b if hist_sum is None else hist_sum + hf_mean_b

                # ---- Future ----
                ff = att.get("future_flags", None)
                if ff is not None and ff.numel() > 0:
                    # (B, dec_len, N_fut)
                    ff_mean_b = ff.mean(dim=0).mean(dim=0)  # (N_fut,)
                    fut_sum = ff_mean_b if fut_sum is None else fut_sum + ff_mean_b

                count += 1
                del xb, att
                torch.cuda.empty_cache()

        results = {}
        if static_sum is not None:
            results["static_importance"] = (static_sum / count).cpu().numpy()
        if hist_sum is not None:
            results["historical_importance"] = (hist_sum / count).cpu().numpy()
        if fut_sum is not None:
            results["future_importance"] = (fut_sum / count).cpu().numpy()

        print("✅ Feature importance computed (batched, memory-safe).")
        return results

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def reset_temp_folder(self):
        print("Resetting temp folder...")
        utils.create_folder_if_not_exist(self._temp_folder)
        if os.path.exists(self._temp_folder):
            shutil.rmtree(self._temp_folder)
        os.makedirs(self._temp_folder, exist_ok=True)

    def save(self, model_folder: str):
        os.makedirs(model_folder, exist_ok=True)
        path = os.path.join(model_folder, f"{self.name}.pt")
        torch.save(self.state_dict(), path)
        print(f"Model saved to: {path}")

    def load(self, folder_name):
        model_path = os.path.join(folder_name, "TemporalFusionTransformer.pt")
        print(f"Model loaded from: {model_path}")
        state = torch.load(model_path, map_location=self.device)

        with torch.no_grad():
            dummy_input = torch.zeros((1, self.time_steps, self.input_size), device=self.device)
            try:
                _ = self(dummy_input)
            except Exception as e:
                print(f"(Warmup forward for submodule init failed non-fatally: {e})")

        missing_keys, unexpected_keys = self.load_state_dict(state, strict=False)
        if unexpected_keys:
            print(f"⚠️ Ignored {len(unexpected_keys)} unexpected keys: e.g. {unexpected_keys[:5]}")
        if missing_keys:
            print(f"⚠️ Missing keys: {missing_keys}")
        print("✅ Model successfully loaded (non-strict).")


    # ------------------------------------------------------------------
    # hyperparameter search
    # ------------------------------------------------------------------
    @classmethod
    def get_hyperparm_choices(cls):
        return {
            "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
            "hidden_layer_size": [10, 20, 40, 80, 160, 240, 320],
            "minibatch_size": [64, 128, 256],
            "learning_rate": [1e-4, 1e-3, 1e-2],
            "max_gradient_norm": [0.01, 1.0, 100.0],
            "num_heads": [1, 4],
            "stack_size": [1],
        }

