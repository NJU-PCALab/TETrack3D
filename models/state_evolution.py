import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.layers import trunc_normal_
except ImportError:
    from torch.nn.init import trunc_normal_

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class StateCompressor(nn.Module):
    def __init__(self, dim, num_slots=2, bottleneck_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.scale = dim ** -0.5

        self.state_queries = nn.Parameter(torch.randn(1, num_slots, dim))
        self.token_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)
        self.geo_norm = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.geo_proj = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.state_queries, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, tokens, centers=None):
        if centers is not None:
            rel_centers = centers - centers.mean(dim=1, keepdim=True)
            tokens = tokens + self.geo_norm(self.geo_proj(rel_centers))
        tokens = self.token_norm(tokens)
        batch_size = tokens.shape[0]
        queries = self.query_norm(self.state_queries.expand(batch_size, -1, -1))

        q = self.q_proj(queries)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn_logits.softmax(dim=-1)

        state_tokens = torch.matmul(attn, v)
        state_tokens = self.out_proj(state_tokens)
        state_tokens = self.out_norm(state_tokens + queries)
        return state_tokens


class TemporalStatePredictor(nn.Module):
    def __init__(
        self,
        state_dim,
        num_slots,
        depth=2,
        expand=2,
        kernel_size=3,
        drop=0.0,
        d_state=16,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba-ssm is required for TemporalStatePredictor. "
                "Install it in the target environment before training."
            )
        self.state_dim = state_dim
        self.num_slots = num_slots
        model_dim = state_dim * num_slots

        self.input_norm = nn.LayerNorm(model_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    norm=nn.LayerNorm(model_dim),
                    mamba=Mamba(
                        d_model=model_dim,
                        d_state=d_state,
                        d_conv=kernel_size,
                        expand=expand,
                    ),
                    drop=nn.Dropout(drop),
                )
            )
            for _ in range(depth)
        ])
        self.output_norm = nn.LayerNorm(model_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for norm in [self.input_norm, self.output_norm]:
            nn.init.constant_(norm.weight, 1.0)
            nn.init.constant_(norm.bias, 0)

        for block in self.blocks:
            nn.init.constant_(block["norm"].weight, 1.0)
            nn.init.constant_(block["norm"].bias, 0)

        for module in self.pred_head.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        final_linear = self.pred_head[-1]
        nn.init.constant_(final_linear.weight, 0)
        nn.init.constant_(final_linear.bias, 0)

    def forward(self, history_states):
        x = history_states.reshape(history_states.shape[0], history_states.shape[1], -1)
        x = self.input_norm(x)
        for block in self.blocks:
            residual = x
            x = block["norm"](x)
            x = block["mamba"](x)
            x = residual + block["drop"](x)
        pred = self.pred_head(self.output_norm(x[:, -1, :]))
        return pred.reshape(history_states.shape[0], self.num_slots, self.state_dim)
