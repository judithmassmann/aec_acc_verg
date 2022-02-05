import math
import numpy as np


class Vec3:
    def __init__(self, x, y, z):
        self._v = np.array([x, y, z], dtype=np.float32)

    @property
    def x(self):
        return self._v[0]

    @x.setter
    def x(self, value):
        self._v[0] = value

    @property
    def y(self):
        return self._v[1]

    @y.setter
    def y(self, value):
        self._v[1] = value

    @property
    def z(self):
        return self._v[2]

    @z.setter
    def z(self, value):
        self._v[2] = value

    def __add__(self, v):
        out = self._v + v._v
        return Vec3(out[0], out[1], out[2])

    def __iadd__(self, v):
        self._v = self._v + v._v
        return self

    def __sub__(self, v):
        out = self._v - v._v
        return Vec3(out[0], out[1], out[2])

    def __isub__(self, v):
        self._v = self._v - v._v
        return self

    def __mul__(self, scalar):
        out = self._v * scalar
        return Vec3(out[0], out[1], out[2])

    def __rmul__(self, scalar):
        out = scalar * self._v
        return Vec3(out[0], out[1], out[2])

    def __truediv__(self, scalar):
        inva = 1.0 / scalar
        out = self._v * inva
        return Vec3(out[0], out[1], out[2])

    def __neg__(self):
        out = -1 * self._v
        return Vec3(out[0], out[1], out[2])

    @staticmethod
    def cross(v1, v2):
        return Vec3(v1.y * v2.z - v1.z * v2.y,
                    v1.z * v2.x - v1.x * v2.z,
                    v1.x * v2.y - v1.y * v2.x)

    def dot(self, v):
        return np.dot(self._v, v._v)

    def normalize(self):
        sqr_length = self.dot(self)
        inv_length = 1 / math.sqrt(sqr_length)
        return inv_length * Vec3(self.x, self.y, self.z)

    def get_data(self):
        return self._v


class Vec4:
    def __init__(self, x, y, z, w):
        self._v = np.array([x, y, z, w], dtype=np.float32)

    @property
    def x(self):
        return self._v[0]

    @x.setter
    def x(self, value):
        self._v[0] = value

    @property
    def y(self):
        return self._v[1]

    @y.setter
    def y(self, value):
        self._v[1] = value

    @property
    def z(self):
        return self._v[2]

    @z.setter
    def z(self, value):
        self._v[2] = value

    @property
    def w(self):
        return self._v[3]

    @w.setter
    def w(self, value):
        self._v[3] = value

    def __add__(self, v):
        out = self._v + v._v
        return Vec4(out[0], out[1], out[2], out[3])

    def __iadd__(self, v):
        self._v = self._v + v._v
        return self

    def __sub__(self, v):
        out = self._v - v._v
        return Vec4(out[0], out[1], out[2], out[3])

    def __isub__(self, v):
        self._v = self._v - v._v
        return self

    def __mul__(self, scalar):
        out = self._v * scalar
        return Vec4(out[0], out[1], out[2], out[3])

    def __rmul__(self, scalar):
        out = scalar * self._v
        return Vec4(out[0], out[1], out[2], out[3])

    def __truediv__(self, scalar):
        inv = 1.0 / scalar
        out = self._v * inv
        return Vec4(out[0], out[1], out[2], out[3])

    def __neg__(self):
        out = -1 * self._v
        return Vec4(out[0], out[1], out[2], out[3])

    def dot(self, v: object):
        return np.dot(self._v, v._v)

    def normalize(self):
        sqr_length = self.dot(self)
        inv_length = 1 / math.sqrt(sqr_length)
        return inv_length * Vec4(self.x, self.y, self.z, self.w)

    def get_data(self):
        return self._v


class Spherical:
    def __init__(self, radius, theta, phi):
        self._v = np.array([radius, theta, phi], dtype=np.float32)

    @property
    def radius(self):
        return self._v[0]

    @radius.setter
    def radius(self, value):
        self._v[0] = value

    @property
    def theta(self):
        return self._v[1]

    @theta.setter
    def theta(self, value):
        self._v[1] = value

    @property
    def phi(self):
        return self._v[2]

    @phi.setter
    def phi(self, value):
        self._v[2] = value

    def to_Vec3(self):
        sin = math.sin
        cos = math.cos

        sp = sin(self.phi)
        st = sin(self.theta)
        cp = cos(self.phi)
        ct = cos(self.theta)

        return Vec3(cp * self.radius * ct,
                    cp * self.radius * st,
                    self.radius * sp)
