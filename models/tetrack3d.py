import torch
from torch import nn

from .localization_head import LocalizationHead
from .recon import RECONBackbone
from .state_evolution import StateCompressor, TemporalStatePredictor


class TETrack3D(nn.Module):
    """TETrack3D with optional training-only state-evolution modules."""

    def __init__(
        self,
        config,
        pretrained_backbone=None,
        enable_training_modules=True,
    ):
        super().__init__()
        self.config = config
        self.use_history = bool(config.backbone.use_history)
        self.backbone = RECONBackbone(config.backbone)
        self.localization_head = LocalizationHead(config.localization)

        state_config = config.state_evolution
        self.enable_training_modules = bool(
            enable_training_modules and state_config.enabled
        )
        if self.enable_training_modules:
            self.state_compressor = StateCompressor(
                dim=self.backbone.feature_dim,
                num_slots=state_config.num_tokens,
            )
            self.temporal_predictor = TemporalStatePredictor(
                state_dim=self.backbone.feature_dim,
                num_slots=state_config.num_tokens,
                depth=state_config.depth,
                expand=state_config.expand,
                kernel_size=state_config.conv_kernel,
                drop=state_config.dropout,
                d_state=state_config.d_state,
            )

        if pretrained_backbone:
            self.pretrained_backbone_report = self.backbone.load_pretrained(
                pretrained_backbone
            )
        else:
            self.pretrained_backbone_report = None

    def encode(self, point_clouds, mask_references, history=None):
        batch_size, num_regions, num_points, _ = point_clouds.shape
        output = self.backbone(
            point_clouds.reshape(batch_size * num_regions, num_points, 3),
            mask_references,
            history=history,
            use_history=self.use_history,
        )
        xyz = output["xyz"]
        indices = output["indices"]
        return {
            "xyz": xyz.reshape(
                batch_size, num_regions, xyz.shape[1], xyz.shape[2]
            ),
            "features": output["features"],
            "indices": indices.reshape(
                batch_size, num_regions, indices.shape[1]
            ),
            "mask_features": output["mask_features"],
            "history": output["history"],
            "search_tokens": output["search_tokens"],
            "search_centers": output["search_centers"],
        }

    def localize(self, features, mask_features, xyz, box_size, center_gt=None):
        inputs = {
            "features": features,
            "mask_features": mask_features,
            "xyz": xyz,
            "box_size": box_size,
        }
        if center_gt is not None:
            inputs["center_gt"] = center_gt
        return self.localization_head(inputs)

    def compress_state(self, search_tokens, search_centers):
        if not self.enable_training_modules:
            raise RuntimeError("State-evolution modules are disabled for inference.")
        return self.state_compressor(search_tokens, search_centers)

    def predict_next_state(self, history_states):
        if not self.enable_training_modules:
            raise RuntimeError("State-evolution modules are disabled for inference.")
        history_length = int(self.config.state_evolution.history_length)
        if history_length > 0:
            history_states = history_states[:, -history_length:]
        return self.temporal_predictor(history_states)
