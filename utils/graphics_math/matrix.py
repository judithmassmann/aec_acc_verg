import utils.graphics_math.vector as vector
import utils.graphics_math.constant_tools as constant
import math
import numpy as np


class Mat3:
    def __init__(self, x, y, z):
        if isinstance(x, vector.Vec3):
            self._m = np.array([x.get_data(),
                                y.get_data(),
                                z.get_data()])
        else:
            self._m = np.array([x, y, z])

    def __add__(self, m):
        out = self._m + m._m
        return Mat3(out[0], out[1], out[2])

    def __iadd__(self, m):
        self._m = self._m + m._m
        return self

    def __sub__(self, m):
        out = self._m - m._m
        return Mat3(out[0], out[1], out[2])

    def __isub__(self, m):
        self._m = self._m - m._m
        return self

    def __mul__(self, value):

        if isinstance(value, Mat3):
            out = np.dot(self._m, value._m)
            return Mat3(out[0], out[1], out[2])
        elif isinstance(value, vector.Vec3):
            out = np.dot(self._m, value._v)
            return vector.Vec3(out[0], out[1], out[2])

        else:
            out = value * self._m
            return Mat3(out[0], out[1], out[2])

    def __rmul__(self, value):

        if isinstance(value, Mat3):
            out = np.dot(value._m, self._m)
            return Mat3(out[0], out[1], out[2])
        elif isinstance(value, vector.Vec3):
            out = np.dot(value._v, self._m)
            return vector.Vec3(out[0], out[1], out[2])

        else:
            out = value * self._m
            return Mat3(out[0], out[1], out[2])

    def __imul__(self, value):
        if isinstance(value, Mat3):
            self._m = np.dot(self._m, value._m)
            return self
        else:
            self._m = self._m * m._m
            return self

    def __div__(self, scalar):
        inva = 1.0 / scalar
        out = self._m * inva
        return Mat3(out[0], out[1], out[2])

    def __rdiv__(self, scalar):
        inva = 1.0 / scalar
        out = inva * self._m
        return Mat3(out[0], out[1], out[2])

    def __getitem__(self, index):
        return self._m[index]

    def to_Mat4(self):
        mat = self._m
        return Mat4([mat[0][0], mat[0][1], mat[0][2], 0.0],
                    [mat[1][0], mat[1][1], mat[1][2], 0.0],
                    [mat[2][0], mat[2][1], mat[2][2], 0.0],
                    [0.0, 0.0, 0.0, 1.0])

    def transponse(self):
        out = self._m.transpose()
        return Mat3(out[0], out[1], out[2])

    def transponse_self(self):
        self._m = self._m.transpose()
        return self

    def inverse(self):
        out = np.linalg.inv(self._m)
        return Mat3(out[0], out[1], out[2])

    def inverse_self(self):
        self._m = np.linalg.inv(self._m)

    def zero(self):
        self._m = np.zeros((3, 3), dtype=np.float32)
        return self

    def indentity(self):
        self._m = np.identity(3, dtype=np.float32)
        return self

    def get_data(self):
        return self._m


class Mat4:
    def __init__(self, x, y, z, w):
        if isinstance(x, vector.Vec4):
            self._m = np.array([x.get_data(),
                                y.get_data(),
                                z.get_data(),
                                w.get_data()])
        else:
            self._m = np.array([x, y, z, w])

    def __add__(self, m):
        out = self._m + m._m
        return Mat4(out[0], out[1], out[2], out[3])

    def __iadd__(self, m):
        self._m = self._m + m._m
        return self

    def __sub__(self, m):
        out = self._m - m._m
        return Mat4(out[0], out[1], out[2], out[3])

    def __isub__(self, m):
        self._m = self._m - m._m
        return self

    def __mul__(self, value):

        if isinstance(value, Mat4):
            out = np.dot(self._m, value._m)
            return Mat4(out[0], out[1], out[2], out[3])
        elif isinstance(value, vector.Vec4):
            out = np.dot(self._m, value._v)
            return vector.Vec4(out[0], out[1], out[2], out[3])

        else:
            out = value * self._m
            return Mat4(out[0], out[1], out[2], out[3])

    def __rmul__(self, value):

        if isinstance(value, Mat4):
            out = np.dot(value._m, self._m)
            return Mat4(out[0], out[1], out[2], out[3])
        elif isinstance(value, vector.Vec4):
            out = np.dot(value._v, self._m)

            return vector.Vec4(out[0], out[1], out[2], out[3])

        else:
            out = value * self._m
            return Mat4(out[0], out[1], out[2], out[3])

    def __imul__(self, value):
        if isinstance(value, Mat3):
            self._m = np.dot(self._m, value._m)
            return self
        else:
            self._m = self._m * self._m
            return self

    def __div__(self, scalar):
        inv = 1.0 / scalar
        out = self._m * inv
        return Mat4(out[0], out[1], out[2], out[3])

    def __rdiv__(self, scalar):
        inv = 1.0 / scalar
        out = inv * self._m
        return Mat4(out[0], out[1], out[2], out[3])

    def __getitem__(self, index):
        return self._m[index]

    def transponse(self):
        out = self._m.transpose()
        return Mat4(out[0], out[1], out[2], out[3])

    def transponse_self(self):
        self._m = self._m.transpose()
        return self

    def inverse(self):
        out = np.linalg.inv(self._m)
        return Mat4(out[0], out[1], out[2], out[3])

    def inverse_self(self):
        self._m = np.linalg.inv(self._m)

    def zero(self):
        self._m = np.zeros((4, 4), dtype=np.float32)
        return self

    def indentity(self):
        self._m = np.identity(4, dtype=np.float32)
        return self

    def get_data(self):
        return self._m
