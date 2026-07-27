import copy
import numpy as np
from pyquaternion import Quaternion
from .point_cloud import PointCloud


def resample_pcd(pcd, n_sample, is_training=True, return_idx=False):
    # random sampling from points
    pcd = pcd.points.T
    num_points = pcd.shape[0]
    new_pts_idx = None
    rng = np.random if is_training else np.random.default_rng(1)
    if num_points >= 1:
        if num_points < n_sample:
            new_pts_idx = rng.choice(
                num_points, size=n_sample-num_points, replace=True)
            idx = np.arange(num_points)
            rng.shuffle(idx)
            new_pts_idx = np.concatenate(
                [idx, new_pts_idx], axis=0)
        elif num_points > n_sample:
            new_pts_idx = rng.choice(
                num_points, size=n_sample, replace=False)
        else:
            new_pts_idx = np.arange(num_points)
            rng.shuffle(new_pts_idx)
    if new_pts_idx is not None:
        pcd = pcd[new_pts_idx, :].copy()
    else:
        pcd = np.zeros((n_sample, 3), dtype='float32')
    pcd = PointCloud(pcd.T)
    if return_idx:
        return pcd, new_pts_idx
    else:
        return pcd


def crop_pcd_axis_aligned(pcd, box, offset=0, scale=1.0, return_mask=False):
    """
    crop the pc using the box in the axis-aligned manner
    """
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = pcd.points[0, :] < maxi[0]
    x_filt_min = pcd.points[0, :] > mini[0]
    y_filt_max = pcd.points[1, :] < maxi[1]
    y_filt_min = pcd.points[1, :] > mini[1]
    z_filt_max = pcd.points[2, :] < maxi[2]
    z_filt_min = pcd.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_pcd = PointCloud(pcd.points[:, close].copy())
    if return_mask:
        return new_pcd, close
    else:
        return new_pcd


def crop_pcd_oriented(pcd, box, offset=0, scale=1.0, return_mask=False):
    """
    crop the pc using the exact box.
    slower than 'crop_pc_axis_aligned' but more accurate
    """

    box_tmp = copy.deepcopy(box)
    new_pcd = PointCloud(pcd.points.copy())
    rot_mat = np.transpose(box_tmp.rotation_matrix)
    trans = -box_tmp.center

    # align data
    new_pcd.translate(trans)
    box_tmp.translate(trans)
    new_pcd.rotate(rot_mat)
    box_tmp.rotate(Quaternion(matrix=rot_mat))

    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = new_pcd.points[0, :] < maxi[0]
    x_filt_min = new_pcd.points[0, :] > mini[0]
    y_filt_max = new_pcd.points[1, :] < maxi[1]
    y_filt_min = new_pcd.points[1, :] > mini[1]
    z_filt_max = new_pcd.points[2, :] < maxi[2]
    z_filt_min = new_pcd.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_pcd = PointCloud(new_pcd.points[:, close])

    # transform back to the original coordinate system
    new_pcd.rotate(np.transpose(rot_mat))
    new_pcd.translate(-trans)
    if return_mask:
        return new_pcd, close
    else:
        return new_pcd


def get_offset_box(box, offset, use_z=True, offset_max=[2.0, 2.0, 1.0], degree=True, is_training=True):
    rot_quat = Quaternion(matrix=box.rotation_matrix)
    trans = np.array(box.center)

    new_box = copy.deepcopy(box)

    new_box.translate(-trans)
    new_box.rotate(rot_quat.inverse)
    if len(offset) == 3:
        use_z = False
    if len(offset) == 3:
        new_box.rotate(
            Quaternion(axis=[0, 0, 1], degrees=offset[2]) if degree else Quaternion(axis=[0, 0, 1], radians=offset[2]))
    elif len(offset) == 4:
        new_box.rotate(
            Quaternion(axis=[0, 0, 1], degrees=offset[3]) if degree else Quaternion(axis=[0, 0, 1], radians=offset[3]))
    if is_training:
        if np.abs(offset[0]) > min(new_box.wlh[0], offset_max[0]):
            offset[0] = np.random.uniform(
                0, min(new_box.wlh[0], offset_max[0])) * np.sign(offset[0])
        if np.abs(offset[1]) > min(new_box.wlh[1], offset_max[1]):
            offset[1] = np.random.uniform(
                0, min(new_box.wlh[1], offset_max[1])) * np.sign(offset[1])
        if use_z and np.abs(offset[2]) > min(new_box.wlh[2], offset_max[2]):
            offset[2] = np.random.uniform(
                0, min(new_box.wlh[2], offset_max[2])) * np.sign(offset[2])
    if use_z:
        new_box.translate(np.array([offset[0], offset[1], offset[2]]))
    else:
        new_box.translate(np.array([offset[0], offset[1], 0]))

    # APPLY PREVIOUS TRANSFORMATION
    new_box.rotate(rot_quat)
    new_box.translate(trans)
    return new_box


def crop_and_center_pcd(pcd, box, offset=0, scale=1.0, offset2=0, normalize=False, return_box=False):
    """
    crop and center the pc using the given box
    """
    new_pcd = crop_pcd_axis_aligned(
        pcd, box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(box)

    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    new_pcd.translate(trans)
    new_box.translate(trans)
    new_pcd.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    # print('HERE!', new_box, offset+offset2, scale)
    new_pcd = crop_pcd_axis_aligned(
        new_pcd, new_box, offset=offset+offset2, scale=scale)
    # print('HERE!', new_pcd.points.T)
    if normalize:
        new_pcd.normalize(box.wlh)
    if return_box:
        return new_pcd, new_box
    else:
        return new_pcd


def merge_template_pcds(pcds, boxes, offset=0, offset2=0, scale=1.0, normalize=False, return_box=False):
    if len(pcds) == 0:
        return PointCloud(np.ones((3, 0)))
    new_pcd = [np.ones((pcds[0].points.shape[0], 0), dtype='float64')]
    for pcd, box in zip(pcds, boxes):
        cropped_pcd, new_box = crop_and_center_pcd(
            pcd, box, offset=offset, offset2=offset2, scale=scale, normalize=normalize, return_box=True)
        # try:
        if cropped_pcd.nbr_points() > 0:
            new_pcd.append(cropped_pcd.points)

    new_pcd = PointCloud(np.concatenate(new_pcd, axis=1))
    if return_box:
        return new_pcd, new_box
    else:
        return new_pcd


def _euclidean_cdist(points_a, points_b):
    """Vectorized Euclidean distance, avoids SciPy dependency."""
    diff = points_a[:, None, :] - points_b[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def get_point_to_box_distance(pcd, box):
    """
    generate the BoxCloud for the given pc and box
    :param pc: Pointcloud object or numpy array
    :param box:
    :return:
    """
    if isinstance(pcd, PointCloud):
        points = pcd.points.T.copy()  # N,3
    else:
        points = pcd.copy()  # N,3
        assert points.shape[1] == 3
    box_corners = box.corners()  # 3,8
    box_centers = box.center.reshape(-1, 1)  # 3,1
    box_points = np.concatenate([box_centers, box_corners], axis=1).T  # 9,3
    p2b_dist = _euclidean_cdist(points, box_points)  # N,9
    return p2b_dist


def get_pcd_in_box_mask(pcd, box, offset=0, scale=1.0):
    """check which points of PC are inside the box"""
    box_tmp = copy.deepcopy(box)
    new_pcd = PointCloud(pcd.points.copy())
    rot_mat = np.transpose(box_tmp.rotation_matrix)
    trans = -box_tmp.center

    # align data
    new_pcd.translate(trans)
    box_tmp.translate(trans)
    new_pcd.rotate(rot_mat)
    box_tmp.rotate(Quaternion(matrix=rot_mat))

    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = new_pcd.points[0, :] < maxi[0]
    x_filt_min = new_pcd.points[0, :] > mini[0]
    y_filt_max = new_pcd.points[1, :] < maxi[1]
    y_filt_min = new_pcd.points[1, :] > mini[1]
    z_filt_max = new_pcd.points[2, :] < maxi[2]
    z_filt_min = new_pcd.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    assert close.shape[0] == new_pcd.points.shape[1]

    return close


def transform_box(box, ref_box):
    new_box = copy.deepcopy(box)
    new_box.translate(-ref_box.center)
    new_box.rotate(Quaternion(matrix=ref_box.rotation_matrix.T))
    return new_box
