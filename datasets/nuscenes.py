import os.path as osp
import pickle as pkl
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
import bisect
import torch
import nuscenes
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import nuscenes.utils.splits

from .utils import *
from .base_dataset import BaseDataset, EvalDatasetWrapper
from utils import pl_ddp_rank


class NuScenesTrackingDataset(BaseDataset):

    cat2n_cat = {
        'void / ignore': ['animal', 'human.pedestrian.personal_mobility', 'human.pedestrian.stroller',
                          'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
                          'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack',
                          'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.construction', 'vehicle.bicycle', 'vehicle.motorcycle'],
        'Bus': ['vehicle.bus.bendy', 'vehicle.bus.rigid'],
        'Car': ['vehicle.car'],
        'Pedestrian': ['human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
                       'human.pedestrian.police_officer'],
        'Trailer': ['vehicle.trailer'],
        'Truck': ['vehicle.truck'],
        'All': ['vehicle.bus.bendy', 'vehicle.bus.rigid','vehicle.car','vehicle.trailer','vehicle.truck','human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
                       'human.pedestrian.police_officer',],
    }
    n_cat2cat = {
        "animal": "void / ignore",
        "human.pedestrian.personal_mobility": "void / ignore",
        "human.pedestrian.stroller": "void / ignore",
        "human.pedestrian.wheelchair": "void / ignore",
        "movable_object.barrier": "void / ignore",
        "movable_object.debris": "void / ignore",
        "movable_object.pushable_pullable": "void / ignore",
        "movable_object.trafficcone": "void / ignore",
        "static_object.bicycle_rack": "void / ignore",
        "vehicle.emergency.ambulance": "void / ignore",
        "vehicle.emergency.police": "void / ignore",
        "vehicle.construction": "void / ignore",
        "vehicle.bicycle": "void / ignore",
        "vehicle.bus.bendy": "Bus",
        "vehicle.bus.rigid": "Bus",
        "vehicle.car": "Car",
        "vehicle.motorcycle": "void / ignore",
        "human.pedestrian.adult": "Pedestrian",
        "human.pedestrian.child": "Pedestrian",
        "human.pedestrian.construction_worker": "Pedestrian",
        "human.pedestrian.police_officer": "Pedestrian",
        "vehicle.trailer": "Trailer",
        "vehicle.truck": "Truck",
    }

    def __init__(self, split_type, cfg, log):
        super().__init__(split_type, cfg, log)

        assert cfg.category_name in [
            'Bus', 'Car', 'Pedestrian', 'Truck', 'Trailer','All']

        self.data_root_dir = cfg.data_root_dir

        self.preload_offset = cfg.train_cfg.preload_offset if split_type == 'train' else cfg.eval_cfg.preload_offset
        self.cache = cfg.train_cfg.cache if split_type == 'train' else cfg.eval_cfg.cache
        self.key_frame_only = True
        self.min_points = 1 if split_type in ['val', 'test'] else -1

        if self.cache:
            if not cfg.debug:
                cache_file_dir = osp.join(
                    self.cfg.data_root_dir, f'NuXcenes_{self.cfg.category_name}_{split_type}_{self.cfg.coordinate_mode}_{self.preload_offset}.cache')
            else:
                cache_file_dir = osp.join(
                    self.cfg.data_root_dir, f'NuXcenes_DEBUG_{self.cfg.category_name}_{split_type}_{self.cfg.coordinate_mode}_{self.preload_offset}.cache')
            if osp.exists(cache_file_dir):
                self.log.info(f'Loading data from cache file {cache_file_dir}')
                with open(cache_file_dir, 'rb') as f:
                    tracklets = pkl.load(f)
                tracklets = self.filter_tracklets(tracklets, split_type)
                self.tracklet_num_frames = [len(tracklet['frames'])
                                            for tracklet in tracklets]
                self.tracklet_st_frame_id = []
                self.tracklet_ed_frame_id = []
                last_ed_frame_id = 0
                for num_frames in self.tracklet_num_frames:
                    assert num_frames > 0
                    self.tracklet_st_frame_id.append(last_ed_frame_id)
                    last_ed_frame_id += num_frames
                    self.tracklet_ed_frame_id.append(last_ed_frame_id)

            else:
                self.nusc = NuScenes(version=self._resolve_version(
                                         ['v1.0-mini', 'v1.0-trainval'] if cfg.debug else ['v1.0-trainval']),
                                     dataroot=cfg.data_root_dir, verbose=False)
                track_instances = self._build_track_instances(
                    split_type, cfg.category_name, self.min_points)

                self.tracklet_annotations = self._build_tracklet_annotations(
                    track_instances)
                self.tracklet_annotations = self.filter_tracklet_annos(
                    self.tracklet_annotations, split_type)
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

                tracklets = []
                for tracklet_id in tqdm(range(len(self.tracklet_annotations)), desc='[%6s]Loading pcds ' % self.split_type.upper(), disable=pl_ddp_rank() != 0):
                    frames = []
                    for frame_anno in self.tracklet_annotations[tracklet_id]:
                        frames.append(self._build_frame(frame_anno))
                    # continue
                    comp_template_pcd = merge_template_pcds(
                        [frame['pcd'] for frame in frames],
                        [frame['bbox'] for frame in frames],
                        offset=cfg.target_offset,
                        scale=cfg.target_scale
                    )
                    assert comp_template_pcd is not None
                    if self.preload_offset > 0:
                        for frame in frames:
                            frame['pcd'] = crop_pcd_axis_aligned(
                                frame['pcd'], frame['bbox'], offset=self.preload_offset)

                    tracklets.append({
                        'comp_template_pcd': comp_template_pcd,
                        'frames': frames
                    })
                # assert False
                with open(cache_file_dir, 'wb') as f:
                    self.log.info(
                        f'Saving data to cache file {cache_file_dir}')
                    pkl.dump(tracklets, f)
            self.tracklets = tracklets
        else:
            if split_type != 'train':
                version = self._resolve_version(
                    ['v1.0-mini', 'v1.0-trainval'] if not cfg.test_version else [cfg.test_version, 'v1.0-trainval']
                )
            else:
                version = self._resolve_version(
                    ['v1.0-mini', 'v1.0-trainval'] if cfg.debug else ['v1.0-trainval']
                )
            self.nusc = NuScenes(version=version,
                                 dataroot=cfg.data_root_dir, verbose=False)
            track_instances = self._build_track_instances(
                split_type, cfg.category_name, self.min_points)

            self.tracklet_annotations = self._build_tracklet_annotations(
                track_instances)
            self.tracklet_annotations = self.filter_tracklet_annos(
                self.tracklet_annotations, split_type)
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

            self.tracklets = None

    def _resolve_version(self, candidates):
        for version in candidates:
            if version and osp.exists(osp.join(self.data_root_dir, version)):
                return version
        raise FileNotFoundError(
            f'NuScenes metadata not found under {self.data_root_dir}. Tried versions: {candidates}'
        )

    def filter_tracklets(self, tracklets, split_type):
        if split_type == 'train':
            min_tracklet_length = max(
                int(getattr(self.cfg, 'tracklet_length_lb', 0)),
                int(getattr(self.cfg, 'num_smp_frames_per_tracklet', 0)),
            )
            return [tracklet for tracklet in tracklets if len(tracklet['frames']) >= min_tracklet_length]
        else:
            return tracklets

    def filter_tracklet_annos(self, tracklet_annos, split_type):
        if split_type == 'train':
            min_tracklet_length = max(
                int(getattr(self.cfg, 'tracklet_length_lb', 0)),
                int(getattr(self.cfg, 'num_smp_frames_per_tracklet', 0)),
            )
            return [tracklet_anno for tracklet_anno in tracklet_annos if len(tracklet_anno) >= min_tracklet_length]
        else:
            return tracklet_annos

    def get_dataset(self):
        if self.split_type == 'train':
            return TrainDatasetWrapper(self, self.cfg, self.log)
        else:
            return EvalDatasetWrapper(self, self.cfg, self.log)

    def num_tracklets(self):
        return len(self.tracklet_annotations)

    def num_frames(self):
        return self.tracklet_ed_frame_id[-1]

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

    def _build_track_instances(self, split_type, category_name, min_points):
        general_classes = self.cat2n_cat[category_name]
        instances = []
        scene_splits = nuscenes.utils.splits.create_splits_scenes()
        for instance in self.nusc.instance:
            anno = self.nusc.get('sample_annotation',
                                 instance['first_annotation_token'])
            sample = self.nusc.get('sample', anno['sample_token'])
            scene = self.nusc.get('scene', sample['scene_token'])
            instance_category = self.nusc.get(
                'category', instance['category_token'])['name']
            if scene['name'] in scene_splits['train_track' if split_type == 'train' else 'val'] and anno['num_lidar_pts'] >= min_points and \
                    (category_name is None or category_name is not None and instance_category in general_classes):
                instances.append(instance)
        return instances

    def _build_tracklet_annotations(self, track_instances):
        tracklet_annotations = []

        for instance in track_instances:
            track_anno = []
            curr_anno_token = instance['first_annotation_token']

            while curr_anno_token != '':

                ann_record = self.nusc.get(
                    'sample_annotation', curr_anno_token)
                sample = self.nusc.get('sample', ann_record['sample_token'])
                sample_data_lidar = self.nusc.get(
                    'sample_data', sample['data']['LIDAR_TOP'])

                curr_anno_token = ann_record['next']
                if self.key_frame_only and not sample_data_lidar['is_key_frame']:
                    continue
                track_anno.append(
                    {"sample_data_lidar": sample_data_lidar, "box_anno": ann_record})

            tracklet_annotations.append(track_anno)
        return tracklet_annotations

    def _build_frame(self, frame_anno):
        sample_data_lidar = frame_anno['sample_data_lidar']
        box_anno = frame_anno['box_anno']
        bbox = BoundingBox(box_anno['translation'], box_anno['size'], Quaternion(box_anno['rotation']),
                           name=box_anno['category_name'])
        pcd_path = osp.join(
            self.data_root_dir, sample_data_lidar['filename'])
        # if osp.exists(pcd_path):
        #     return {"pcd": None, "bbox": None, 'anno': None}
        # else:
        #     print(pcd_path)
        #     return {"pcd": None, "bbox": None, 'anno': None}
        pcd = LidarPointCloud.from_file(pcd_path)

        cs_record = self.nusc.get(
            'calibrated_sensor', sample_data_lidar['calibrated_sensor_token'])
        pcd.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pcd.translate(np.array(cs_record['translation']))

        poserecord = self.nusc.get(
            'ego_pose', sample_data_lidar['ego_pose_token'])
        pcd.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pcd.translate(np.array(poserecord['translation']))

        pcd = PointCloud(points=pcd.points)
        return {"pcd": pcd, "bbox": bbox, 'anno': frame_anno}


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
            template_pcd, template_bbox_ref = crop_and_center_pcd(
                pcd,
                bbox,
                offset=self.cfg.frame_offset,
                offset2=self.cfg.frame_offset2,
                scale=self.cfg.frame_scale,
                return_box=True
            )
            template_mask_ref = get_pcd_in_box_mask(
                template_pcd, template_bbox_ref).astype(np.float32)
            template_mask_ref[template_mask_ref == 0] = 0.2
            template_mask_ref[template_mask_ref == 1] = 0.8

            bbox = transform_box(bbox, base_bbox)

            if i == 0:
                pcd, bbox_ref = crop_and_center_pcd(
                    pcd,
                    base_bbox,
                    offset=self.cfg.frame_offset,
                    offset2=self.cfg.frame_offset2,
                    scale=self.cfg.frame_scale,
                    return_box=True
                )
                mask_ref = get_pcd_in_box_mask(
                    pcd, bbox_ref).astype(np.float32)
                bc_ref = get_point_to_box_distance(pcd, bbox_ref)
                bc_gt = get_point_to_box_distance(pcd, bbox)
                seg_label = get_pcd_in_box_mask(pcd, bbox).astype(int)
                mask_ref[mask_ref == 0] = 0.2
                mask_ref[mask_ref == 1] = 0.8
            else:
                pcd = crop_and_center_pcd(
                    pcd,
                    base_bbox,
                    offset=self.cfg.frame_offset,
                    offset2=self.cfg.frame_offset2,
                    scale=self.cfg.frame_scale
                )
                mask_ref = np.ones_like(
                    get_pcd_in_box_mask(pcd, bbox, scale=1.25).astype(np.float32)) * 0.5
                bc_ref = np.zeros((pcd.points.shape[1], 9), dtype=np.float32)
                bc_gt = get_point_to_box_distance(pcd, bbox)
                seg_label = get_pcd_in_box_mask(pcd, bbox).astype(int)
            mask_gt = get_pcd_in_box_mask(pcd, bbox, scale=1.25).astype(np.float32)
            bbox_gt = np.array([bbox.center[0], bbox.center[1], bbox.center[2], (
                bbox.orientation.degrees if self.cfg.degree else bbox.orientation.radians) * bbox.orientation.axis[-1]])

            assert pcd.nbr_points() >= 5
            assert template_pcd.nbr_points() >= 5

            pcd, idx = resample_pcd(
                pcd, self.cfg.frame_npts, return_idx=True, is_training=True)
            mask_gt = mask_gt[idx]
            mask_ref = mask_ref[idx]
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
            else:
                template_pcd, template_idx = resample_pcd(
                    template_pcd, self.cfg.frame_npts, return_idx=True, is_training=True)
                template_pcds.append(template_pcd.points.T)
                template_mask_refs.append(template_mask_ref[template_idx])
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
            bc_refs=bc_refs,
            bc_gts=bc_gts,
            bbox_gts=bbox_gts,
            first_mask_gt=first_mask_gt,
            first_bbox_gt=first_bbox_gt,
            is_dynamic_gts=is_dynamic_gts,
            seg_labels=seg_labels,
            template_pcds=template_pcds,
            template_mask_refs=template_mask_refs
        )
        return self._to_float_tensor(data)

    def _to_float_tensor(self, data):
        tensor_data = {}
        for k, v in data.items():
            tensor_data[k] = torch.FloatTensor(np.array(v))
        return tensor_data

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
