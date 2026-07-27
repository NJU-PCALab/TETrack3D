import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from utils.metrics import estimateAccuracy, estimateOverlap, estimateWaymoOverlap


def _trapezoid(values, x):
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(values, x=x)
    return np.trapz(values, x=x)


def success_score(overlaps):
    overlaps = np.asarray(overlaps, dtype=np.float64)
    if overlaps.size == 0:
        return 0.0
    thresholds = np.linspace(0.0, 1.0, 21)
    curve = np.asarray([(overlaps >= threshold).mean() for threshold in thresholds])
    return float(_trapezoid(curve, x=thresholds) * 100.0)


def precision_score(accuracies):
    accuracies = np.asarray(accuracies, dtype=np.float64)
    if accuracies.size == 0:
        return 0.0
    thresholds = np.linspace(0.0, 2.0, 21)
    curve = np.asarray([(accuracies <= threshold).mean() for threshold in thresholds])
    return float(_trapezoid(curve, x=thresholds) * 50.0)


class Evaluator:
    def __init__(self, config, tracker, log, output_dir, save_predictions=False):
        self.config = config
        self.tracker = tracker
        self.log = log
        self.output_dir = Path(output_dir)
        self.save_predictions = save_predictions

    def _dataset_kind(self):
        name = str(self.config.dataset.name).lower()
        if 'nuscenes' in name:
            return 'nuscenes'
        if 'waymo' in name:
            return 'waymo'
        if 'kitti' in name:
            return 'kitti'
        raise ValueError(f'Unsupported dataset type: {name}')

    @staticmethod
    def _kitti_category(tracklet):
        anno = tracklet[0].get('anno') if tracklet else None
        if anno is None:
            return None
        category = anno.get('type')
        return str(category) if category is not None else None

    @staticmethod
    def _nuscenes_category(tracklet):
        anno = tracklet[0].get('anno', {}) if tracklet else {}
        box_anno = anno.get('box_anno', {}) if isinstance(anno, dict) else {}
        category = str(box_anno.get('category_name', ''))
        mappings = {
            'vehicle.bus': 'Bus',
            'vehicle.car': 'Car',
            'human.pedestrian': 'Pedestrian',
            'vehicle.trailer': 'Trailer',
            'vehicle.truck': 'Truck',
        }
        for prefix, display_name in mappings.items():
            if category.startswith(prefix):
                return display_name
        return None

    def _evaluate_tracklet(self, tracklet, kind):
        if self.tracker.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred_bboxes, gt_bboxes = self.tracker.track(tracklet)
        if self.tracker.device.type == 'cuda':
            torch.cuda.synchronize()
        runtime = time.perf_counter() - start

        overlaps = []
        accuracies = []
        for frame_id, (prediction, target) in enumerate(zip(pred_bboxes, gt_bboxes)):
            if kind == 'nuscenes':
                anno = tracklet[frame_id].get('anno', {})
                sample_data = anno.get('sample_data_lidar', anno) if isinstance(anno, dict) else {}
                if int(sample_data.get('is_key_frame', 1)) != 1:
                    continue
            if kind == 'waymo':
                overlap = estimateWaymoOverlap(
                    target, prediction, dim=self.config.evaluation.iou_space
                )
            else:
                overlap = estimateOverlap(
                    target,
                    prediction,
                    dim=self.config.evaluation.iou_space,
                    up_axis=self.config.dataset.up_axis,
                )
            accuracy = estimateAccuracy(
                target,
                prediction,
                dim=self.config.evaluation.iou_space,
                up_axis=self.config.dataset.up_axis,
            )
            overlaps.append(overlap)
            accuracies.append(accuracy)
        return overlaps, accuracies, runtime, pred_bboxes

    def run(self, dataset, max_tracklets=None, start_tracklet=0):
        kind = self._dataset_kind()
        start_tracklet = int(start_tracklet)
        if start_tracklet < 0 or start_tracklet >= len(dataset):
            raise ValueError(
                f'start_tracklet must be in [0, {len(dataset) - 1}], '
                f'got {start_tracklet}'
            )
        stop_tracklet = (
            len(dataset)
            if max_tracklets is None
            else min(len(dataset), start_tracklet + int(max_tracklets))
        )
        limit = stop_tracklet - start_tracklet
        all_overlaps = []
        all_accuracies = []
        total_runtime = 0.0
        total_frames = 0
        category_values = defaultdict(lambda: {'overlaps': [], 'accuracies': [], 'runtime': 0.0})
        waymo_values = defaultdict(lambda: {'success_sum': 0.0, 'precision_sum': 0.0, 'weight': 0})
        predictions = []

        for shard_index, tracklet_id in enumerate(
            range(start_tracklet, stop_tracklet)
        ):
            tracklet = dataset[tracklet_id]
            if kind == 'nuscenes':
                anno = tracklet[0].get('anno', {}) if tracklet else {}
                box_anno = anno.get('box_anno', anno) if isinstance(anno, dict) else {}
                if not tracklet or int(box_anno.get('num_lidar_pts', 1)) == 0:
                    continue

            overlaps, accuracies, runtime, pred_bboxes = self._evaluate_tracklet(tracklet, kind)
            all_overlaps.extend(overlaps)
            all_accuracies.extend(accuracies)
            total_runtime += runtime
            total_frames += len(tracklet)

            if kind == 'kitti':
                category = self._kitti_category(tracklet)
            elif kind == 'nuscenes':
                category = self._nuscenes_category(tracklet)
            else:
                category = None
            if category:
                category_values[category]['overlaps'].extend(overlaps)
                category_values[category]['accuracies'].extend(accuracies)
                category_values[category]['runtime'] += runtime

            if kind == 'waymo':
                mode = tracklet[0]['mode']
                weight = max(len(tracklet) - 1, 1)
                values = waymo_values[mode]
                values['success_sum'] += success_score(overlaps) * weight
                values['precision_sum'] += precision_score(accuracies) * weight
                values['weight'] += weight

            if self.save_predictions:
                predictions.append([bbox.encode() for bbox in pred_bboxes])

            self.log.info(
                'Tracklet %d/%d (dataset id %d): Precision=%.3f Success=%.3f Frames=%d Runtime=%.4fs',
                shard_index + 1,
                limit,
                tracklet_id,
                precision_score(all_accuracies),
                success_score(all_overlaps),
                total_frames,
                total_runtime,
            )

        summary = {
            'dataset': kind,
            'precision': precision_score(all_accuracies),
            'success': success_score(all_overlaps),
            'runtime_per_frame': total_runtime / max(total_frames, 1),
            'n_frames': total_frames,
            'n_tracklets': limit,
            'start_tracklet': start_tracklet,
        }
        if category_values:
            summary['per_category'] = {
                category: {
                    'precision': precision_score(values['accuracies']),
                    'success': success_score(values['overlaps']),
                    'runtime': values['runtime'],
                    'n_frames': len(values['overlaps']),
                }
                for category, values in sorted(category_values.items())
            }
        if kind == 'waymo':
            splits = {}
            for mode, values in sorted(waymo_values.items()):
                weight = max(values['weight'], 1)
                splits[mode] = {
                    'precision': values['precision_sum'] / weight,
                    'success': values['success_sum'] / weight,
                    'weight': values['weight'],
                }
            total_weight = sum(item['weight'] for item in waymo_values.values())
            if total_weight:
                summary['precision'] = sum(item['precision_sum'] for item in waymo_values.values()) / total_weight
                summary['success'] = sum(item['success_sum'] for item in waymo_values.values()) / total_weight
            summary['splits'] = splits

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / 'test_summary.json').open('w') as handle:
            json.dump(summary, handle, indent=2)
        if self.save_predictions:
            with (self.output_dir / 'result.json').open('w') as handle:
                json.dump(predictions, handle)
        self.log.info('Final summary: %s', json.dumps(summary, indent=2))
        return summary
