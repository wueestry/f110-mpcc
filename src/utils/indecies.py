from enum import IntEnum


class State(IntEnum):
    POS_ON_CENTER_LINE_S = 0
    MIN_DIST_TO_CENTER_LINE_N = 1
    ORIENTATION_ALPHA = 2
    VELOCITY_V = 3
    DUTY_CYCLE_D = 4
    STEERING_ANGLE_DELTA = 5
    PROGRESS_THETA = 6


class Input(IntEnum):
    D_DUTY_CYCLE = 0
    D_STEERING_ANGLE = 1
    D_PROGRESS = 2


class Parameter(IntEnum):
    m = 0
    C1 = 1
    C2 = 2
    Cm1 = 3
    Cm2 = 4
    Cr0 = 5
    Cr2 = 6
    Cr3 = 7
    Iz = 8
    lr = 9
    lf = 10
    Bf = 11
    Cf = 12
    Df = 13
    Br = 14
    Cr = 15
    Dr = 16
    Imax_c = 17
    Caccel = 18
    qc = 19
    ql = 20
    gamma = 21
    r1 = 22
    r2 = 23
    r3 = 24
