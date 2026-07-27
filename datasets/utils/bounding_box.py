import numpy as np
from pyquaternion import Quaternion


class BoundingBox:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self, center, size, orientation, label=np.nan, score=np.nan, velocity=(np.nan, np.nan, np.nan),
                 name=None):
        """
        :param center: [<float>: 3]. Center of box given as x, y, z.
        :param size: [<float>: 3]. Size of box in width, length, height.
        :param orientation: <Quaternion>. Box orientation.
        :param label: <int>. Integer label, optional.
        :param score: <float>. Classification score, optional.
        :param velocity: [<float>: 3]. Box velocity in x, y, z direction.
        :param name: <str>. Box name, optional. Can be used e.g. for denote category name.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        # assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(
            self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (
            np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (
            np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}, name: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1], self.center[2], self.wlh[0],
                               self.wlh[1], self.wlh[2], self.orientation.axis[0], self.orientation.axis[1],
                               self.orientation.axis[2], self.orientation.degrees, self.orientation.radians,
                               self.velocity[0], self.velocity[1], self.velocity[2], self.name)

    def encode(self):
        """
        Encodes the box instance to a JSON-friendly vector representation.
        :return: [<float>: 16]. List of floats encoding the box.
        """
        return self.center.tolist() + self.wlh.tolist() + self.orientation.elements.tolist() + [self.label] + [self.score] + self.velocity.tolist() + [self.name]

    @classmethod
    def decode(cls, data):
        """
        Instantiates a Box instance from encoded vector representation.
        :param data: [<float>: 16]. Output from encode.
        :return: <Box>.
        """
        return BoundingBox(data[0:3], data[3:6], Quaternion(data[6:10]), label=data[10], score=data[11], velocity=data[12:15],
                           name=data[15])

    @property
    def rotation_matrix(self):
        """
        Return a rotation matrix.
        :return: <np.float: (3, 3)>.
        """
        return self.orientation.rotation_matrix

    def translate(self, x):
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        :return: <None>.
        """
        self.center += x

    def rotate(self, quaternion):
        """
        Rotates box.
        :param quaternion: <Quaternion>. Rotation to apply.
        :return: <None>.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def transform(self, transf_matrix):
        transformed = np.dot(transf_matrix[0:3, 0:4].T, self.center)
        self.center = transformed[0:3]/transformed[3]
        self.orientation = self.orientation * \
            Quaternion(matrix=transf_matrix[0:3, 0:3])
        self.velocity = np.dot(transf_matrix[0:3, 0:3], self.velocity)

    def corners(self, wlh_factor=1.0):
        """
        Returns the bounding box corners.
        :param wlh_factor: <float>. Multiply w, l, h by a factor to inflate or deflate the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self):
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]
