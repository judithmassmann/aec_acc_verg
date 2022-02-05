import utils.graphics_math.constant_tools as constant
import utils.graphics_math.matrix as matrix
import math
import numpy as np


class Angles:
    def __init__(self, pitch, yaw, roll):
        self._ang = np.array([pitch, yaw, roll], dtype=np.float32)

    @property
    def pitch(self):
        return self._ang[0]

    @pitch.setter
    def pitch(self, value):
        self._ang[0] = value

    @property
    def yaw(self):
        return self._ang[1]

    @yaw.setter
    def yaw(self, value):
        self._ang[1] = value

    @property
    def roll(self):
        return self._ang[2]

    @roll.setter
    def roll(self, value):
        self._ang[2] = value

    def __add__(self, ang):
        out = self._ang + ang._ang
        return Angles(out[0], out[1], out[2])

    def __iadd__(self, ang):
        self._ang = self._ang + ang._ang
        return self

    def __sub__(self, ang):
        out = self._ang - ang._ang
        return Angles(out[0], out[1], out[2])

    def __isub__(self, v):
        self._v = self._v - v._v
        return self

    def __mul__(self, scalar):
        out = self._ang * scalar
        return Angles(out[0], out[1], out[2])

    def __rmul__(self, scalar):
        out = scalar * self._ang
        return Angles(out[0], out[1], out[2])

    def __truediv__(self, scalar):
        inva = 1.0 / scalar
        out = self._ang * inva
        return Angles(out[0], out[1], out[2])

    def to_Mat3(self):
        sin = math.sin
        cos = math.cos

        sp = sin(self.pitch * constant.M_DEG2RAD)
        sy = sin(self.yaw * constant.M_DEG2RAD)
        sr = sin(self.roll * constant.M_DEG2RAD)

        cp = cos(self.pitch * constant.M_DEG2RAD)
        cy = cos(self.yaw * constant.M_DEG2RAD)
        cr = cos(self.roll * constant.M_DEG2RAD)

        mat =  matrix.Mat3([cp * cy, cp * sy, -sp],
                                 [sr * sp * cy + cr * -sy, sr * sp * sy + cr * cy, sr * cp],
                                 [cr * sp * cy + -sr * -sy, cr * sp * sy + -sr * cy, cr * cp])

        return mat

    def to_Mat4(self):
        return self.to_Mat3().to_Mat4()

    def get_data(self):
        return self._ang
