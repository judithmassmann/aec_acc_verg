import math

PI = math.pi
TWO_PI = 2.0 * PI
M_DEG2RAD = PI / 180.0
M_RAD2DEG = 180.0 / PI


class HMS:
    RAD_H = TWO_PI / 24
    RAD_M = RAD_H / 60
    RAD_S = RAD_M / 60


class GMS:
    RAD_G = TWO_PI / 360
    RAD_M = RAD_G / 60
    RAD_S = RAD_M / 60


def deg_to_rad(deg):
    return deg * (PI / 180)


def rad_to_deg(rad):
    return (180 / PI) * rad


def hms_to_rad(h, m, s):
    return HMS.RAD_H * h + HMS.RAD_M * m + HMS.RAD_S * s


def sgms_to_rad(sign, g, m, s):
    expr = GMS.RAD_G * g + GMS.RAD_M * m + GMS.RAD_S * s
    if sign == "+":
        return expr
    else:
        return -expr


def sec_to_deg(sec):
    return GMS.RAD_S * sec


def sec_to_day(sec):
    YEAR_PERIOD = 365.2425
    SEC_TO_DAY = 60 * 60 * 24
    return (sec / SEC_TO_DAY) / YEAR_PERIOD
