import utils.graphics_math.matrix as matrix
import utils.graphics_math.vector as vector
import utils.graphics_math.angles as angles
import utils.graphics_math.constant_tools as constant
import math


class Transform:
    @staticmethod
    def scale(vec3):
        mat = matrix.Mat4([vec3.x, 0.0, 0.0, 0.0],
                          [0.0, vec3.y, 0.0, 0.0],
                          [0.0, 0.0, vec3.z, 0.0],
                          [0.0, 0.0, 0.0, 1.0])
        return mat

    @staticmethod
    def translate(vec3):
        mat = matrix.Mat4([1.0, 0.0, 0.0, vec3.x],
                          [0.0, 1.0, 0.0, vec3.y],
                          [0.0, 0.0, 1.0, vec3.z],
                          [0.0, 0.0, 0.0, 1.0])
        return mat

    @staticmethod
    def rotate(vec3):
        mat = angles.Angles(vec3.x, vec3.y, vec3.z)
        return mat.to_Mat4()

    @staticmethod
    def ortho(left, right, bottom, top, z_near, z_far):
        result = matrix.Mat4([1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0])

        result[0][0] = 2.0 / (right - left)
        result[1][1] = 2.0 / (top - bottom)
        result[3][0] = -((right + left) / (right - left))
        result[3][1] = -((top + bottom) / (top - bottom))
        result[2][2] = -(1.0 / (z_far - z_near))
        result[3][2] = -(z_near / (z_far - z_near))
        return result

    @staticmethod
    def perspective(fov, aspect, z_near, z_far):
        fov = fov * constant.M_DEG2RAD

        f = 1.0 / math.tan(fov / 2.0)
        a = (z_far + z_near) / (z_far - z_near)
        b = (2.0 * z_far * z_near) / (z_far - z_near)

        result = matrix.Mat4([0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0])

        result[0][0] = f / aspect
        result[1][1] = f
        result[2][2] = -a
        result[2][3] = -b
        result[3][2] = -1.0

        return result

    @staticmethod
    def transform_basis(eye, center, up):
        f = (center - eye).normalize()
        s = vector.Vec3.cross(f, up).normalize()
        u = vector.Vec3.cross(s, f)

        result = matrix.Mat3([1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0])

        result[0][0] = s.x
        result[0][1] = s.y
        result[0][2] = s.z
        result[1][0] = u.x
        result[1][1] = u.y
        result[1][2] = u.z
        result[2][0] = -f.x
        result[2][1] = -f.y
        result[2][2] = -f.z

        return result
