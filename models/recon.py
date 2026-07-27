from pathlib import Path

import torch
from timm.layers import DropPath
from torch import nn
from torch.nn import functional as F

from . import layers


class PointGroupEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, output_dim, 1),
        )

    def forward(self, point_groups):
        batch_size, num_groups, group_size, _ = point_groups.shape
        points = point_groups.reshape(batch_size * num_groups, group_size, 3)
        local_features = self.first_conv(points.transpose(1, 2))
        global_features = local_features.max(dim=2, keepdim=True).values
        features = torch.cat(
            (global_features.expand_as(local_features), local_features), dim=1
        )
        features = self.second_conv(features).max(dim=2).values
        return features.reshape(batch_size, num_groups, self.output_dim)


def pairwise_squared_distance(source, target):
    distance = -2 * torch.matmul(source, target.transpose(1, 2))
    distance += (source**2).sum(dim=-1, keepdim=True)
    distance += (target**2).sum(dim=-1).unsqueeze(1)
    return distance


class PointGrouping(nn.Module):
    def __init__(self, num_groups, group_size):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size

    def forward(self, points):
        batch_size, num_points, _ = points.shape
        if num_points < self.num_groups:
            raise ValueError(
                f"Expected at least {self.num_groups} points, got {num_points}."
            )

        center_indices = torch.arange(
            self.num_groups, device=points.device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)
        centers = torch.gather(
            points,
            dim=1,
            index=center_indices.unsqueeze(-1).expand(-1, -1, points.shape[-1]),
        ).contiguous()
        neighborhood_indices = pairwise_squared_distance(centers, points).topk(
            self.group_size, dim=-1, largest=False, sorted=False
        ).indices
        batch_offsets = (
            torch.arange(batch_size, device=points.device).view(-1, 1, 1)
            * num_points
        )
        flat_indices = (neighborhood_indices + batch_offsets).reshape(-1)
        neighborhoods = points.reshape(batch_size * num_points, 3)[flat_indices]
        neighborhoods = neighborhoods.reshape(
            batch_size, self.num_groups, self.group_size, 3
        )
        neighborhoods = neighborhoods - centers.unsqueeze(2)
        # RECON uses the group-center indices for its point-mask projection;
        # the KNN indices are only needed to construct local neighborhoods.
        return neighborhoods.contiguous(), centers, center_indices


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        output = self.dropout(self.activation(self.fc1(inputs)))
        return self.dropout(self.fc2(output))


class HistoricalAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attention_dropout,
        projection_dropout,
        history_length,
        fixed_history,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Feature dimension must be divisible by num_heads.")
        if fixed_history < 0 or fixed_history > history_length:
            raise ValueError("fixed_history must be within the history length.")

        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.attention_dropout = attention_dropout
        self.history_length = history_length
        self.fixed_history = fixed_history
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def _update_history(self, previous, current_key, current_value):
        current = {
            "key": current_key.contiguous(),
            "value": current_value.contiguous(),
        }
        if previous is None:
            return current

        patch_length = current_key.shape[2]
        fixed_tokens = patch_length * self.fixed_history
        total_tokens = patch_length * self.history_length
        recent_tokens = total_tokens - fixed_tokens

        previous_key = previous["key"]
        previous_value = previous["value"]
        fixed_key = previous_key[:, :, :fixed_tokens]
        fixed_value = previous_value[:, :, :fixed_tokens]
        recent_key = torch.cat(
            (previous_key[:, :, fixed_tokens:], current_key), dim=2
        )
        recent_value = torch.cat(
            (previous_value[:, :, fixed_tokens:], current_value), dim=2
        )
        if recent_tokens > 0:
            recent_key = recent_key[:, :, -recent_tokens:]
            recent_value = recent_value[:, :, -recent_tokens:]
        else:
            recent_key = recent_key[:, :, :0]
            recent_value = recent_value[:, :, :0]
        return {
            "key": torch.cat((fixed_key, recent_key), dim=2).contiguous(),
            "value": torch.cat((fixed_value, recent_value), dim=2).contiguous(),
        }

    @staticmethod
    def _causal_frame_mask(key_length, patch_length, device):
        if key_length % patch_length != 0:
            raise ValueError("Historical tokens must contain complete frame slots.")
        num_key_frames = key_length // patch_length
        if num_key_frames < 2:
            raise ValueError("Template-search attention requires two current frames.")
        key_frames = torch.arange(num_key_frames, device=device)
        query_frames = torch.arange(num_key_frames - 2, num_key_frames, device=device)
        frame_mask = key_frames.unsqueeze(0) <= query_frames.unsqueeze(1)
        return frame_mask.repeat_interleave(patch_length, dim=0).repeat_interleave(
            patch_length, dim=1
        )

    def forward(self, inputs, history=None, use_history=True):
        batch_size, num_tokens, dim = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch_size, num_tokens, 3, self.num_heads, dim // self.num_heads
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        patch_length = num_tokens // 2

        updated_history = None
        if use_history:
            updated_history = self._update_history(
                history,
                key[:, :, :patch_length],
                value[:, :, :patch_length],
            )
            if history is not None:
                key = torch.cat((history["key"], key), dim=2)
                value = torch.cat((history["value"], value), dim=2)

        attention_mask = self._causal_frame_mask(
            key.shape[2], patch_length, query.device
        )
        output = F.scaled_dot_product_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scale,
        )
        output = output.transpose(1, 2).reshape(batch_size, num_tokens, dim)
        return self.proj_drop(self.proj(output)), updated_history


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        dropout,
        drop_path,
        history_length,
        fixed_history,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = HistoricalAttention(
            dim=dim,
            num_heads=num_heads,
            attention_dropout=dropout,
            projection_dropout=dropout,
            history_length=history_length,
            fixed_history=fixed_history,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, hidden_dim=dim * 4, dropout=dropout)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, inputs, history=None, use_history=True):
        attention, updated_history = self.attn(
            self.norm1(inputs), history=history, use_history=use_history
        )
        output = inputs + self.drop_path1(attention)
        output = output + self.drop_path2(self.mlp(self.norm2(output)))
        return output, updated_history


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        dropout,
        drop_path_rate,
        history_length,
        fixed_history,
    ):
        super().__init__()
        drop_paths = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                drop_path=drop_paths[index],
                history_length=history_length,
                fixed_history=fixed_history,
            )
            for index in range(depth)
        )

    def forward(self, inputs, history=None, use_history=True):
        if history is None:
            history = [None] * len(self.blocks)
        if len(history) != len(self.blocks):
            raise ValueError("Historical memory depth does not match the backbone.")

        output = inputs
        updated_history = []
        for block, block_history in zip(self.blocks, history):
            output, block_history = block(
                output, history=block_history, use_history=use_history
            )
            updated_history.append(block_history)
        return output, updated_history if use_history else None


class RECONBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_dim = int(config.feature_dim)
        self.num_groups = int(config.num_groups)
        self.use_history = bool(config.use_history)

        self.group_divider = PointGrouping(
            num_groups=self.num_groups,
            group_size=int(config.group_size),
        )
        self.encoder = PointGroupEncoder(output_dim=self.feature_dim)
        self.learn = nn.Parameter(torch.ones(1, self.num_groups))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.feature_dim),
        )
        self.blocks = TransformerEncoder(
            dim=self.feature_dim,
            depth=int(config.depth),
            num_heads=int(config.num_heads),
            dropout=float(config.dropout),
            drop_path_rate=float(config.drop_path_rate),
            history_length=int(config.history_length),
            fixed_history=int(config.fixed_history),
        )
        self.norm = nn.LayerNorm(self.feature_dim)
        self.fc_mask = (
            layers.Seq(self.feature_dim)
            .conv1d(self.feature_dim, bn=True)
            .conv1d(self.feature_dim, bn=True)
            .conv1d(self.feature_dim, bn=True)
            .conv1d(128, activation=None)
        )
        self.fc1 = layers.Seq(self.feature_dim).conv1d(128, activation=None)
        self.mask_emb = layers.Seq(1).conv1d(
            self.feature_dim, activation=None
        )

    def load_pretrained(self, checkpoint_path):
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("base_model", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError(f"Invalid RECON checkpoint: {path}")

        normalized = {}
        for raw_key, value in state_dict.items():
            key = raw_key.removeprefix("module.")
            for prefix in (
                "MAE_encoder.",
                "ACT_encoder.",
                "transformer_k.",
                "base_model.",
            ):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            normalized[key] = value

        current = self.state_dict()
        compatible = {
            key: value
            for key, value in normalized.items()
            if key in current and current[key].shape == value.shape
        }
        if not compatible:
            raise RuntimeError(f"No RECON tensors matched the backbone: {path}")
        self.load_state_dict(compatible, strict=False)
        return len(compatible), len(normalized) - len(compatible)

    def forward(
        self,
        points,
        mask_references,
        history=None,
        use_history=None,
    ):
        neighborhoods, centers, indices = self.group_divider(points)
        mask_references = mask_references.reshape(-1, mask_references.shape[-1])
        mask_references = torch.gather(
            mask_references, dim=1, index=indices.long()
        ).float()
        mask_references = mask_references * self.learn.expand(
            mask_references.shape[0], -1
        )
        mask_embeddings = self.mask_emb(mask_references.unsqueeze(1)).transpose(
            1, 2
        )

        tokens = self.encoder(neighborhoods) + mask_embeddings
        tokens = tokens.reshape(
            -1, 2 * tokens.shape[-2], tokens.shape[-1]
        )
        positions = self.pos_embed(centers).reshape_as(tokens)
        use_history = self.use_history if use_history is None else use_history
        tokens, updated_history = self.blocks(
            tokens + positions,
            history=history,
            use_history=use_history,
        )
        tokens = self.norm(tokens)

        search_tokens = tokens[:, -self.num_groups:]
        search_features = search_tokens.transpose(1, 2)
        search_centers = centers.reshape(
            -1, 2, self.num_groups, centers.shape[-1]
        )[:, 1]
        return {
            "xyz": centers,
            "features": self.fc1(search_features),
            "indices": indices.long(),
            "mask_features": self.fc_mask(search_features),
            "history": updated_history,
            "search_tokens": search_tokens,
            "search_centers": search_centers,
        }
