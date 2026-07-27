import time

import pytorch_lightning as pl
import torch
from torchmetrics import Metric

from models import TETrack3D
from tracker import TETrack3DTracker
from utils.metrics import estimateAccuracy, estimateOverlap

from .losses import (
    box_regression_loss,
    center_regression_loss,
    mask_loss,
    objectness_loss,
    state_evolution_loss,
    temporal_distribution_alignment_loss,
)


class TrackingCurveMetric(Metric):
    def __init__(self, thresholds, comparison):
        super().__init__()
        self.register_buffer("thresholds", torch.as_tensor(thresholds).float())
        self.comparison = comparison
        self.add_state(
            "counts",
            default=torch.zeros(len(thresholds)),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.zeros(()), dist_reduce_fx="sum")

    def update(self, values):
        values = values.float().reshape(-1, 1)
        thresholds = self.thresholds.reshape(1, -1)
        if self.comparison == "ge":
            matches = values >= thresholds
        else:
            matches = values <= thresholds
        self.counts += matches.sum(dim=0)
        self.total += values.shape[0]

    def compute(self):
        curve = self.counts / self.total.clamp_min(1)
        return torch.trapz(curve, self.thresholds) * (
            100.0 / (self.thresholds[-1] - self.thresholds[0])
        )


class TETrack3DTrainingModule(pl.LightningModule):
    def __init__(self, config, pretrained_backbone=None, text_logger=None):
        super().__init__()
        self.config = config
        self.text_logger = text_logger
        self.model = TETrack3D(
            config.model,
            pretrained_backbone=pretrained_backbone,
            enable_training_modules=True,
        )
        if self.text_logger is not None and self.model.pretrained_backbone_report:
            loaded, ignored = self.model.pretrained_backbone_report
            self.text_logger.info(
                "Loaded RECON backbone pretraining: %d tensors matched, "
                "%d checkpoint tensors ignored (%s)",
                loaded,
                ignored,
                pretrained_backbone,
            )
        self.tracker = None
        self.validation_success = TrackingCurveMetric(
            torch.linspace(0.0, 1.0, 21), comparison="ge"
        )
        self.validation_precision = TrackingCurveMetric(
            torch.linspace(0.0, 2.0, 21), comparison="le"
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.optimization.learning_rate),
            weight_decay=float(self.config.optimization.weight_decay),
            betas=tuple(self.config.optimization.betas),
            eps=float(self.config.optimization.eps),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.config.training.epochs),
            eta_min=float(self.config.optimization.minimum_learning_rate),
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        point_clouds = batch["pcds"]
        mask_targets = batch["mask_gts"]
        search_mask_references = batch["mask_refs"]
        box_targets = batch["bbox_gts"]
        template_clouds = batch["template_pcds"]
        template_mask_references = batch["template_mask_refs"]
        box_size = batch["lwh"]

        num_steps = point_clouds.shape[1] - 1
        if num_steps < 2:
            raise ValueError("State evolution training requires at least three frames.")

        feature_history = None
        state_history = []
        predicted_next_state = None
        previous_tokens = None
        previous_weights = None
        totals = {
            "mask": point_clouds.new_zeros(()),
            "center_objectness": point_clouds.new_zeros(()),
            "box_objectness": point_clouds.new_zeros(()),
            "center": point_clouds.new_zeros(()),
            "box": point_clouds.new_zeros(()),
            "state": point_clouds.new_zeros(()),
            "alignment": point_clouds.new_zeros(()),
        }
        state_pairs = 0
        alignment_pairs = 0

        for frame_index in range(1, point_clouds.shape[1]):
            template_index = frame_index - 1
            input_masks = torch.cat(
                (
                    template_mask_references[:, [template_index]],
                    search_mask_references[:, [frame_index]],
                ),
                dim=1,
            )
            input_points = torch.cat(
                (
                    template_clouds[:, [template_index]],
                    point_clouds[:, [frame_index]],
                ),
                dim=1,
            )
            encoded = self.model.encode(
                point_clouds=input_points,
                mask_references=input_masks,
                history=feature_history,
            )
            feature_history = encoded["history"]

            search_indices = encoded["indices"][:, 1]
            current_weights = torch.gather(
                mask_targets[:, frame_index], 1, search_indices
            ).float()
            current_tokens = encoded["search_tokens"]
            if previous_tokens is not None:
                totals["alignment"] += temporal_distribution_alignment_loss(
                    previous_tokens,
                    current_tokens,
                    previous_weights,
                    current_weights,
                )
                alignment_pairs += 1
            previous_tokens = current_tokens
            previous_weights = current_weights

            observed_state = self.model.compress_state(
                current_tokens, encoded["search_centers"]
            )
            if predicted_next_state is not None:
                totals["state"] += state_evolution_loss(
                    predicted_next_state, observed_state
                )
                state_pairs += 1
            state_history.append(observed_state)
            max_history = int(self.config.model.state_evolution.history_length)
            if max_history > 0:
                state_history = state_history[-max_history:]
            if frame_index < point_clouds.shape[1] - 1:
                predicted_next_state = self.model.predict_next_state(
                    torch.stack(state_history, dim=1)
                )

            localized = self.model.localize(
                features=encoded["features"],
                mask_features=encoded["mask_features"],
                xyz=encoded["xyz"][:, 1],
                box_size=box_size,
                center_gt=box_targets[:, frame_index, :3],
            )
            foreground = current_weights
            totals["mask"] += mask_loss(localized["mask_logits"], foreground)

            center_targets = box_targets[:, frame_index, :3].unsqueeze(1)
            center_offsets = center_targets - encoded["xyz"][:, 1]
            totals["center"] += center_regression_loss(
                localized["offset_center"],
                center_offsets,
                box_size,
                foreground,
            )

            center_distance = torch.linalg.vector_norm(
                localized["center"] - center_targets, dim=-1
            )
            center_labels = (center_distance < 0.3).float()
            center_valid = torch.ones_like(center_labels)
            totals["center_objectness"] += objectness_loss(
                localized["center_objectness_logits"],
                center_labels,
                center_valid,
            )

            proposal_distance = torch.linalg.vector_norm(
                localized["proposal_centers"] - center_targets, dim=-1
            )
            proposal_labels = (proposal_distance < 0.3).float()
            totals["box_objectness"] += objectness_loss(
                localized["boxes"][:, :, 4],
                proposal_labels,
                torch.ones_like(proposal_labels),
            )
            box_target = box_targets[:, frame_index].unsqueeze(1).expand_as(
                localized["boxes"][:, :, :4]
            )
            totals["box"] += box_regression_loss(
                localized["boxes"][:, :, :4], box_target, proposal_labels
            )

        for name in ("mask", "center_objectness", "box_objectness", "center", "box"):
            totals[name] = totals[name] / num_steps
        totals["state"] = totals["state"] / max(state_pairs, 1)
        totals["alignment"] = totals["alignment"] / max(alignment_pairs, 1)

        weights = self.config.loss
        total_loss = (
            weights.mask * totals["mask"]
            + weights.center_objectness * totals["center_objectness"]
            + weights.box_objectness * totals["box_objectness"]
            + weights.center * totals["center"]
            + weights.box * totals["box"]
            + weights.state_evolution * totals["state"]
            + weights.temporal_alignment * totals["alignment"]
        )
        batch_size = point_clouds.shape[0]
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log_dict(
            {f"train/{name}": value for name, value in totals.items()},
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return total_loss

    def on_validation_epoch_start(self):
        self.validation_success.reset()
        self.validation_precision.reset()
        self.tracker = TETrack3DTracker(
            model=self.model,
            config=self.config,
            device=self.device,
        )

    def validation_step(self, batch, batch_idx):
        tracklet = batch[0]
        start = time.perf_counter()
        predicted_boxes, target_boxes = self.tracker.track(tracklet)
        runtime = time.perf_counter() - start
        overlaps = [
            estimateOverlap(
                target,
                prediction,
                dim=self.config.evaluation.iou_space,
                up_axis=self.config.dataset.up_axis,
            )
            for prediction, target in zip(predicted_boxes, target_boxes)
        ]
        distances = [
            estimateAccuracy(
                target,
                prediction,
                dim=self.config.evaluation.iou_space,
                up_axis=self.config.dataset.up_axis,
            )
            for prediction, target in zip(predicted_boxes, target_boxes)
        ]
        self.validation_success.update(
            torch.as_tensor(overlaps, device=self.device)
        )
        self.validation_precision.update(
            torch.as_tensor(distances, device=self.device)
        )
        if self.text_logger is not None and (batch_idx + 1) % 20 == 0:
            self.text_logger.info(
                "Validation tracklet %d: frames=%d runtime=%.3fs",
                batch_idx + 1,
                len(tracklet),
                runtime,
            )

    def on_validation_epoch_end(self):
        success = self.validation_success.compute()
        precision = self.validation_precision.compute()
        self.log("val_success", success, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        if self.text_logger is not None:
            self.text_logger.info(
                "Validation epoch %d: Success=%.3f Precision=%.3f",
                self.current_epoch,
                float(success.cpu()),
                float(precision.cpu()),
            )
