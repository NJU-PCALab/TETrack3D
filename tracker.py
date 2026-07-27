import numpy as np
import torch

from datasets.utils import (
    crop_and_center_pcd,
    get_offset_box,
    get_pcd_in_box_mask,
    resample_pcd,
)


class TETrack3DTracker:
    def __init__(self, model, config, device, amp_dtype=None):
        self.model = model
        self.config = config
        self.device = device
        self.amp_dtype = amp_dtype

    def _autocast(self):
        enabled = self.device.type == 'cuda' and self.amp_dtype is not None
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=enabled,
        )

    @torch.inference_mode()
    def track(self, tracklet):
        pred_bboxes = []
        gt_bboxes = []
        feature_history = None
        box_size = None
        last_bbox_offset = np.zeros(4, dtype=np.float32)

        for frame_id, frame in enumerate(tracklet):
            previous_id = max(frame_id - 1, 0)
            gt_bboxes.append(frame['bbox'])

            if frame_id == 0:
                base_bbox = frame['bbox']
                box_size = np.asarray(
                    [base_bbox.wlh[1], base_bbox.wlh[0], base_bbox.wlh[2]],
                    dtype=np.float32,
                )
                pred_bboxes.append(frame['bbox'])
                continue

            base_bbox = pred_bboxes[-1]
            search_pcd = crop_and_center_pcd(
                frame['pcd'],
                base_bbox,
                offset=self.config.dataset.frame_offset,
                offset2=self.config.dataset.frame_offset2,
                scale=self.config.dataset.frame_scale,
            )
            template_pcd, template_bbox_ref = crop_and_center_pcd(
                tracklet[previous_id]['pcd'],
                base_bbox,
                offset=self.config.dataset.frame_offset,
                offset2=self.config.dataset.frame_offset2,
                scale=self.config.dataset.frame_scale,
                return_box=True,
            )

            if search_pcd.nbr_points() <= 1:
                pred_bboxes.append(
                    get_offset_box(
                        pred_bboxes[-1],
                        last_bbox_offset,
                        use_z=self.config.dataset.test.use_z,
                        is_training=False,
                    )
                )
                continue

            search_pcd, _ = resample_pcd(
                search_pcd,
                self.config.dataset.num_points,
                is_training=False,
                return_idx=True,
            )
            template_pcd, _ = resample_pcd(
                template_pcd,
                self.config.dataset.num_points,
                is_training=False,
                return_idx=True,
            )

            search_mask_ref = np.full(search_pcd.points.shape[1], 0.5)
            template_mask_ref = get_pcd_in_box_mask(
                template_pcd,
                template_bbox_ref,
                scale=1.25,
            ).astype(np.float32)
            if frame_id != 1:
                template_mask_ref[template_mask_ref == 0] = 0.2
                template_mask_ref[template_mask_ref == 1] = 0.8

            mask_refs = torch.from_numpy(
                np.stack((template_mask_ref, search_mask_ref), axis=0)
            ).unsqueeze(0).to(self.device)
            pcds = torch.from_numpy(
                np.stack((template_pcd.points.T, search_pcd.points.T), axis=0)
            ).unsqueeze(0).float().to(self.device)

            with self._autocast():
                embedded = self.model.encode(
                    point_clouds=pcds,
                    mask_references=mask_refs,
                    history=feature_history,
                )
                feature_history = embedded["history"]
                localized = self.model.localize(
                    features=embedded["features"],
                    mask_features=embedded["mask_features"],
                    xyz=embedded["xyz"][:, 1],
                    box_size=torch.from_numpy(box_size).unsqueeze(0).to(self.device),
                )

            mask_confidence = localized["mask_logits"].sigmoid().max().item()
            proposals = localized["boxes"].squeeze(0).float().cpu().numpy()
            proposals[np.isnan(proposals)] = -1e6
            bbox_offset = proposals[proposals[:, 4].argmax(), :4]
            if mask_confidence < self.config.tracking.missing_threshold:
                bbox_offset = last_bbox_offset
            else:
                last_bbox_offset = bbox_offset
            pred_bboxes.append(
                get_offset_box(
                    pred_bboxes[-1],
                    bbox_offset,
                    use_z=self.config.dataset.test.use_z,
                    is_training=False,
                )
            )

        return pred_bboxes, gt_bboxes
