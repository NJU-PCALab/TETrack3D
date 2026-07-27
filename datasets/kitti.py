import os.path as osp
import pandas as pd
import pickle as pkl
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
import bisect
import torch

from .utils import *
from .base_dataset import BaseDataset, EvalDatasetWrapper
from utils import pl_ddp_rank


class KITTITrackingDataset(BaseDataset):

    def __init__(self, split_type, cfg, log):
        super().__init__(split_type, cfg, log)

        assert cfg.category_name in ['Van', 'Car', 'Pedestrian', 'Cyclist','All']

        if not cfg.debug:
            split_type_to_scene_ids = dict(
                train=list(range(0, 17)),
                val=list(range(17, 19)),
                test=list(range(19, 21))
            )
        else:
            split_type_to_scene_ids = dict(
                train=[0],
                val=[18],
                test=[19]
            )

        self.preload_offset = cfg.train_cfg.preload_offset if split_type == 'train' else cfg.eval_cfg.preload_offset
        self.cache = cfg.train_cfg.cache if split_type == 'train' else cfg.eval_cfg.cache
        self.calibration_info = {}

        scene_ids = split_type_to_scene_ids[split_type]
        self.tracklet_annotations = self._build_tracklet_annotations(scene_ids)

        if self.cache:
            if not cfg.debug:
                cache_file_dir = osp.join(
                    self.cfg.data_root_dir, f'KITTI_{self.cfg.category_name}_{split_type}_{self.cfg.coordinate_mode}_{self.preload_offset}.cache')
            else:
                cache_file_dir = osp.join(
                    self.cfg.data_root_dir, f'KITTI_DEBUG_{self.cfg.category_name}_{split_type}_{self.cfg.coordinate_mode}_{self.preload_offset}.cache')
            if osp.exists(cache_file_dir):
                self.log.info(f'Loading data from cache file {cache_file_dir}')
                with open(cache_file_dir, 'rb') as f:
                    tracklets = pkl.load(f)
            else:
                tracklets = []
                for tracklet_id in tqdm(range(len(self.tracklet_annotations)), desc='[%6s]Loading pcds ' % self.split_type.upper(), disable=pl_ddp_rank() != 0):
                    frames = []
                    for frame_anno in self.tracklet_annotations[tracklet_id]:
                        frames.append(self._build_frame(frame_anno))

                    comp_template_pcd = merge_template_pcds(
                        [frame['pcd'] for frame in frames],
                        [frame['bbox'] for frame in frames],
                        offset=cfg.target_offset,
                        scale=cfg.target_scale
                    )
                    if self.preload_offset > 0:
                        for frame in frames:
                            frame['pcd'] = crop_pcd_axis_aligned(
                                frame['pcd'], frame['bbox'], offset=self.preload_offset)

                    tracklets.append({
                        'comp_template_pcd': comp_template_pcd,
                        'frames': frames
                    })

                with open(cache_file_dir, 'wb') as f:
                    self.log.info(
                        f'Saving data to cache file {cache_file_dir}')
                    pkl.dump(tracklets, f)
            self.tracklets = tracklets
        else:
            self.tracklets = None

        if split_type == 'train':
            self.tracklet_annotations, self.tracklets = self.filter_tracklet(
                self.tracklet_annotations, self.tracklets, cfg.num_smp_frames_per_tracklet)

        self.tracklet_num_frames = [len(tracklet_anno)
                                    for tracklet_anno in self.tracklet_annotations]
        self.tracklet_st_frame_id = []
        self.tracklet_ed_frame_id = []
        last_ed_frame_id = 0
        for num_frames in self.tracklet_num_frames:
            assert num_frames > 0
            self.tracklet_st_frame_id.append(last_ed_frame_id)
            last_ed_frame_id += num_frames
            self.tracklet_ed_frame_id.append(last_ed_frame_id)

    def filter_tracklet(self, t_annos, ts, min_tracklet_length=0):
        keep = [
            index
            for index, annotations in enumerate(t_annos)
            if len(annotations) >= min_tracklet_length
        ]
        filtered_annotations = [t_annos[index] for index in keep]
        filtered_tracklets = None if ts is None else [ts[index] for index in keep]
        return filtered_annotations, filtered_tracklets

    def get_dataset(self):
        if self.split_type == 'train':
            return TrainDatasetWrapper(self, self.cfg, self.log)
        else:
            return EvalDatasetWrapper(self, self.cfg, self.log)

    def num_frames(self):
        return self.tracklet_ed_frame_id[-1]

    def num_tracklets(self):
        return len(self.tracklet_annotations)

    def num_tracklet_frames(self, tracklet_id):
        return self.tracklet_num_frames[tracklet_id]

    def get_frame(self, tracklet_id, frame_id):
        if self.tracklets:
            frame = self.tracklets[tracklet_id]['frames'][frame_id]
            return frame
        else:
            frame_anno = self.tracklet_annotations[tracklet_id][frame_id]
            frame = self._build_frame(frame_anno)
            if self.preload_offset > 0:
                frame['pcd'] = crop_pcd_axis_aligned(
                    frame['pcd'], frame['bbox'], offset=self.preload_offset)
            return frame

    def get_frame_bbox(self, tracklet_id, frame_id):
        if self.tracklets:
            frame = self.tracklets[tracklet_id]['frames'][frame_id]
        else:
            frame_anno = self.tracklet_annotations[tracklet_id][frame_id]
            frame = self._build_frame(frame_anno, bbox_only=True)
        return frame['bbox']

    def get_comp_pcd(self, tracklet_id):
        comp_template_pcd = self.tracklets[tracklet_id]['comp_template_pcd']
        return comp_template_pcd

    def get_tracklet_frame_id(self, idx):
        tracklet_id = bisect.bisect_right(
            self.tracklet_ed_frame_id, idx)
        assert self.tracklet_st_frame_id[
            tracklet_id] <= idx and idx < self.tracklet_ed_frame_id[tracklet_id]
        frame_id = idx - \
            self.tracklet_st_frame_id[tracklet_id]
        return tracklet_id, frame_id

    def _build_tracklet_annotations(self, scene_ids):
        tracklet_annotations = []
        for scene_id in tqdm(scene_ids, desc='[%6s]Loading annos' % self.split_type.upper(), disable=pl_ddp_rank() != 0):
            annotation_file_dir = osp.join(
                self.cfg.data_root_dir, 'label_02/%04d.txt' % scene_id)
            data = pd.read_csv(
                annotation_file_dir,
                sep=' ',
                names=[
                    'frame', 'track_id', 'type', 'truncated', 'occluded',
                    'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
                    'height', 'width', 'length', 'x', 'y', 'z', 'rotation_y'
                ]
            )
            if self.cfg.category_name =='All':
                data = data[(data["type"] == 'Car') |
                            (data["type"] == 'Van') |
                            (data["type"] == 'Pedestrian') |
                            (data["type"] == 'Cyclist')]
            else:
                data = data[data["type"] == self.cfg.category_name]
            data.insert(loc=0, column='scene', value=scene_id)
            track_ids = sorted(data.track_id.unique())
            for track_id in track_ids:
                tracklet_anno = data[data['track_id'] == track_id]
                tracklet_anno = tracklet_anno.sort_values(by=['frame'])
                tracklet_anno = tracklet_anno.reset_index(drop=True)
                tracklet_anno = [frame_anno for _,
                                 frame_anno in tracklet_anno.iterrows()]
                tracklet_annotations.append(tracklet_anno)
        return tracklet_annotations

    @staticmethod
    def _read_calibration_file(filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        return data

    def _build_frame(self, frame_anno, bbox_only=False):
        scene_id = frame_anno['scene']
        frame_id = frame_anno['frame']
        if scene_id in self.calibration_info:
            calib = self.calibration_info[scene_id]
        else:
            calib = self._read_calibration_file(
                osp.join(self.cfg.data_root_dir, 'calib/%04d.txt' % scene_id))
            self.calibration_info[scene_id] = calib
        velo_to_cam = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))

        assert self.cfg.coordinate_mode in ['camera', 'velodyne']

        if self.cfg.coordinate_mode == 'camera':
            bbox_center = [frame_anno["x"], frame_anno["y"] -
                           frame_anno["height"] / 2, frame_anno["z"]]
            size = [frame_anno["width"],
                    frame_anno["length"], frame_anno["height"]]
            orientation = Quaternion(
                axis=[0, 1, 0], radians=frame_anno["rotation_y"]) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
            bbox = BoundingBox(bbox_center, size, orientation)
        else:
            box_center_cam = np.array(
                [frame_anno["x"], frame_anno["y"] - frame_anno["height"] / 2, frame_anno["z"], 1])
            # transform bb from camera coordinate into velo coordinates
            box_center_velo = np.dot(
                np.linalg.inv(velo_to_cam), box_center_cam)
            box_center_velo = box_center_velo[:3]
            size = [frame_anno["width"],
                    frame_anno["length"], frame_anno["height"]]
            orientation = Quaternion(
                axis=[0, 0, -1], radians=frame_anno["rotation_y"]) * Quaternion(axis=[0, 0, -1], degrees=90)
            bbox = BoundingBox(box_center_velo, size, orientation)
        if bbox_only:
            return {'bbox': bbox, 'anno': frame_anno}
        try:
            pcd_file_dir = osp.join(
                self.cfg.data_root_dir, 'velodyne/%04d' % scene_id, '%06d.bin' % frame_id)
            pcd = PointCloud(np.fromfile(
                pcd_file_dir, dtype=np.float32).reshape(-1, 4).T)
            if self.cfg.coordinate_mode == 'camera':
                pcd.transform(velo_to_cam)
        except:
            # in case the Point cloud is missing
            # (0001/[000177-000180].bin)
            pcd = PointCloud(np.array([[0, 0, 0]]).T)

        return {'pcd': pcd, 'bbox': bbox, 'anno': frame_anno}


def print_np(**kwargs):
    for k, v in kwargs.items():
        print(k, np.concatenate((v[:5], v[-5:]), axis=0))


class TrainDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: BaseDataset, cfg, log):
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg
        self.log = log

    def __len__(self):
        # return 100
        return self.dataset.num_frames()

    def _generate_item(self, comp_template_pcd, frames, prev_frame_bboxes):

        frame_pcds = [f['pcd'] for f in frames]
        frame_bboxes = [f['bbox'] for f in frames]

        # if self.cfg.use_seq_aug:
        #     frame_pcds, frame_bboxes = sequence_augment3d(
        #         frame_pcds, frame_bboxes)

        pcds = []
        mask_gts = []
        mask_refs = []
        bbox_gts = []
        bc_refs = []
        bc_gts = []
        seg_labels = []
        is_dynamic_gts = []
        template_pcds = []
        template_mask_refs = []
        wlh = None
        for i, (bbox, pcd, prev_bbox) in enumerate(zip(frame_bboxes, frame_pcds, prev_frame_bboxes)):
            if self.cfg.train_cfg.use_z:
                if self.cfg.use_smp_aug:
                    bbox_offset = np.random.uniform(low=-0.3, high=0.3, size=4)
                    bbox_offset[3] = bbox_offset[3] * \
                        (5 if self.cfg.degree else np.deg2rad(5))
                else:
                    bbox_offset = np.zeros(4)
            else:
                if self.cfg.use_smp_aug:
                    bbox_offset = np.random.uniform(low=-0.3, high=0.3, size=3)
                    bbox_offset[2] = bbox_offset[2] * \
                        (5 if self.cfg.degree else np.deg2rad(5))
                else:
                    bbox_offset = np.zeros(3)

            if i == 0:
                # for the first frame, crop the pcd with the given gt bbox
                base_bbox = bbox
                wlh = bbox.wlh
            else:
                # for other frames, crop the pcd with previous pred bbox
                base_bbox = get_offset_box(
                    prev_bbox, bbox_offset, use_z=self.cfg.train_cfg.use_z, offset_max=self.cfg.offset_max, degree=self.cfg.degree, is_training=True)
                # template_bbox = copy.deepcopy(bbox)
                # template_bbox = transform_box(bbox, template_bbox)
            template_pcd,template_bbox_ref = crop_and_center_pcd(pcd, bbox, offset=self.cfg.frame_offset,
                                          offset2=self.cfg.frame_offset2, scale=self.cfg.frame_scale,return_box=True)
            template_mask_ref = get_pcd_in_box_mask(
                template_pcd, template_bbox_ref).astype(np.float32)
            template_mask_ref[template_mask_ref == 0] = 0.2
            template_mask_ref[template_mask_ref == 1] = 0.8

            bbox = transform_box(bbox, base_bbox)
            
            if i==0:
                pcd,template_bbox_ref = crop_and_center_pcd(pcd, base_bbox, offset=self.cfg.frame_offset,
                                          offset2=self.cfg.frame_offset2, scale=self.cfg.frame_scale,return_box=True)

                mask_ref = get_pcd_in_box_mask(
                    pcd, template_bbox_ref).astype(np.float32)
                bc_ref = get_point_to_box_distance(pcd, template_bbox_ref)
                bc_gt = get_point_to_box_distance(pcd, bbox)
                seg_label = get_pcd_in_box_mask(pcd, bbox).astype(int)
                mask_ref[mask_ref == 0] = 0.2
                mask_ref[mask_ref == 1] = 0.8
                # template_pcd = pcd
                # template_mask_ref = mask_ref
            else:
                pcd = crop_and_center_pcd(pcd, base_bbox, offset=self.cfg.frame_offset,
                                          offset2=self.cfg.frame_offset2, scale=self.cfg.frame_scale)
                mask_ref=np.ones_like(get_pcd_in_box_mask(pcd, bbox,scale=1.25).astype(np.float32)) * 0.5
                bc_ref = np.zeros((pcd.points.shape[1], 9), dtype=np.float32)
                bc_gt = get_point_to_box_distance(pcd, bbox)
                seg_label = get_pcd_in_box_mask(pcd, bbox).astype(int)
            mask_gt = get_pcd_in_box_mask(pcd, bbox,scale=1.25).astype(np.float32)
            bbox_gt = np.array([bbox.center[0], bbox.center[1], bbox.center[2], (
                bbox.orientation.degrees if self.cfg.degree else bbox.orientation.radians) * bbox.orientation.axis[-1]])

            assert pcd.nbr_points() >= 5
            assert template_pcd.nbr_points() >= 5

            pcd, idx = resample_pcd(
                pcd, self.cfg.frame_npts, return_idx=True, is_training=True)
            mask_ref = mask_ref[idx]
            mask_gt = mask_gt[idx]
            bc_ref = bc_ref[idx]
            bc_gt = bc_gt[idx]
            seg_label = seg_label[idx]
            bc_refs.append(bc_ref)
            bc_gts.append(bc_gt)
            seg_labels.append(seg_label)
            mask_refs.append(mask_ref)
            pcds.append(pcd.points.T)
            mask_gts.append(mask_gt)
            bbox_gts.append(bbox_gt)
            if i == 0:
                is_dynamic_gts.append(False)
                template_mask_refs.append(template_mask_ref[idx])
                template_pcds.append(pcd.points.T)
                if template_mask_ref[idx] is None:
                    import pdb
                    pdb.set_trace()
            else:
                template_pcd, template_idx = resample_pcd(template_pcd, self.cfg.frame_npts, return_idx=True, is_training=True)
                template_pcds.append(template_pcd.points.T)
                template_mask_refs.append(template_mask_ref[template_idx])
                # if template_idx is None:
                #     import pdb
                #     pdb.set_trace()
                if np.linalg.norm(bbox_gts[i][:3]-bbox_gts[i-1][:3], ord=2) > self.cfg.dynamic_threshold:
                    is_dynamic_gts.append(True)
                else:
                    is_dynamic_gts.append(False)

        first_mask_gt = mask_gts[0]
        first_bbox_gt = bbox_gts[0]


        data = dict(
            whl=np.array([wlh[0], wlh[2], wlh[1]]),
            lwh=np.array([wlh[1], wlh[0], wlh[2]]),
            pcds=pcds,
            mask_gts=mask_gts,
            mask_refs=mask_refs,
            bc_refs = bc_refs,
            bc_gts = bc_gts,
            bbox_gts=bbox_gts,
            first_mask_gt=first_mask_gt,
            first_bbox_gt=first_bbox_gt,
            is_dynamic_gts=is_dynamic_gts,
            seg_labels = seg_labels,
            template_pcds = template_pcds,
            template_mask_refs = template_mask_refs
        )
        return self._to_float_tensor(data)


    def _to_float_tensor(self, data):
        return {
            key: torch.as_tensor(np.asarray(value), dtype=torch.float32)
            for key, value in data.items()
        }

    def __getitem__(self, idx):

        tracklet_id, frame_id = self.dataset.get_tracklet_frame_id(idx)
        tracklet_length = self.dataset.num_tracklet_frames(tracklet_id)
        assert tracklet_length >= self.cfg.num_smp_frames_per_tracklet

        frame_ids = [frame_id]
        acceptable_set = set(range(
            max(0, frame_id-self.cfg.max_frame_dis),
            min(tracklet_length, frame_id+self.cfg.max_frame_dis+1)
        )).difference(set(frame_ids))
        while len(frame_ids) < self.cfg.num_smp_frames_per_tracklet:
            idx = np.random.choice(list(acceptable_set))
            frame_ids.append(idx)
            new_set = set(range(max(0, idx-self.cfg.max_frame_dis),
                          min(tracklet_length, idx+self.cfg.max_frame_dis+1)))
            acceptable_set = acceptable_set.union(
                new_set).difference(set(frame_ids))

        frame_ids = sorted(frame_ids)
        if np.random.rand() < 0.5:
            # Reverse time
            frame_ids = frame_ids[::-1]
            prev_frame_ids = [min(f_id+1, tracklet_length-1)
                              for f_id in frame_ids]
        else:
            prev_frame_ids = [max(f_id-1, 0)
                              for f_id in frame_ids]

        frames = [self.dataset.get_frame(tracklet_id, id) for id in frame_ids]
        prev_frame_bboxes = [self.dataset.get_frame_bbox(
            tracklet_id, id) for id in prev_frame_ids]
        comp_template_pcd = self.dataset.get_comp_pcd(tracklet_id)

        try:
            return self._generate_item(comp_template_pcd, frames, prev_frame_bboxes)
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]
