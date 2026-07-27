import numpy as np
import torch


class PointCloud:

    def __init__(self, points):
        """
        Class for manipulating and viewing point clouds.
        :param points: <np.float: 4, n>. Input point cloud matrix.
        """
        self.points = points
        if self.points.shape[0] > 3:
            self.points = self.points[0:3, :]

    @staticmethod
    def load_pcd_bin(file_name):
        """
        Loads from binary format. Data is stored as (x, y, z, intensity, ring index).
        :param file_name: <str>.
        :return: <np.float: 4, n>. Point cloud matrix (x, y, z, intensity).
        """
        scan = np.fromfile(file_name, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, :4]
        return points.T

    @classmethod
    def from_file(cls, file_name):
        """
        Instantiate from a .pcl, .pdc, .npy, or .bin file.
        :param file_name: <str>. Path of the pointcloud file on disk.
        :return: <PointCloud>.
        """

        if file_name.endswith('.bin'):
            points = cls.load_pcd_bin(file_name)
        elif file_name.endswith('.npy'):
            points = np.load(file_name)
        else:
            raise ValueError('Unsupported filetype {}'.format(file_name))

        return cls(points)

    def nbr_points(self):
        """
        Returns the number of points.
        :return: <int>. Number of points.
        """
        return self.points.shape[1]

    def subsample(self, ratio):
        """
        Sub-samples the pointcloud.
        :param ratio: <float>. Fraction to keep.
        :return: <None>.
        """
        selected_ind = np.random.choice(np.arange(0, self.nbr_points()),
                                        size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, radius):
        """
        Removes point too close within a certain radius from origin.
        :param radius: <float>.
        :return: <None>.
        """

        x_filt = np.abs(self.points[0, :]) < radius
        y_filt = np.abs(self.points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        self.points = self.points[:, not_close]

    def translate(self, x):
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        :return: <None>.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix):
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        :return: <None>.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix):
        """
        Applies a homogeneous transform.
        :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
        :return: <None>.
        """
        self.points[:3, :] = transf_matrix.dot(
            np.vstack((self.points[:3, :], np.ones(self.nbr_points()))))[:3, :]

    def convertToPytorch(self):
        """
        Helper from pytorch.
        :return: Pytorch array of points.
        """
        return torch.from_numpy(self.points)

    @staticmethod
    def fromPytorch(cls, pytorchTensor):
        """
        Loads from binary format. Data is stored as (x, y, z, intensity, ring index).
        :param pyttorchTensor: <Tensor>.
        :return: <np.float: 4, n>. Point cloud matrix (x, y, z, intensity).
        """
        points = pytorchTensor.numpy()
        # points = points.reshape((-1, 5))[:, :4]
        return cls(points)

    def normalize(self, wlh):
        normalizer = [wlh[1], wlh[0], wlh[2]]
        self.points = self.points / np.atleast_2d(normalizer).T
