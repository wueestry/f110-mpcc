#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# author: Daniel Kloeser

import types

import numpy as np
from acados_template import MX, Function, cos, interpolant, mod, sin, vertcat
from casadi import *


def bicycle_model(s0: list, kapparef: list, d_left: list, d_right: list, cfg_dict: dict):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "Spatialbicycle_model"

    length = len(s0)
    pathlength = s0[-1]
    # copy loop to beginning and end

    s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    s0 = np.append([s0[: length - 1] - s0[length - 1]], s0)
    kapparef = np.append(kapparef, kapparef[1:length])
    kapparef = np.append([kapparef[: length - 1] - kapparef[length - 1]], kapparef)

    d_left = np.append(d_left, d_left[1:length])
    d_left = np.append([d_left[: length - 1] - d_left[length - 1]], d_left)
    d_right = np.append(d_right, d_right[1:length])
    d_right = np.append([d_right[: length - 1] - d_right[length - 1]], d_right)

    N = cfg_dict["N"]

    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)
    outer_bound_s = interpolant("outer_bound_s", "bspline", [s0], d_left)
    inner_bound_s = interpolant("inner_bound_s", "bspline", [s0], d_right)

    ## CasADi Model
    # set up states & controls
    s = MX.sym("s")
    n = MX.sym("n")
    alpha = MX.sym("alpha")
    v = MX.sym("v")
    D = MX.sym("D")
    delta = MX.sym("delta")
    theta = MX.sym("theta")

    x = vertcat(s, n, alpha, v, D, delta, theta)

    # controls
    derD = MX.sym("derD")
    derDelta = MX.sym("derDelta")
    derTheta = MX.sym("derTheta")
    u = vertcat(derD, derDelta, derTheta)

    next_D = D + derD / N
    next_delta = delta + derDelta / N

    # xdot
    sdot = MX.sym("sdot")
    ndot = MX.sym("ndot")
    alphadot = MX.sym("alphadot")
    vdot = MX.sym("vdot")
    Ddot = MX.sym("Ddot")
    deltadot = MX.sym("deltadot")
    thetadot = MX.sym("thetadot")
    xdot = vertcat(sdot, ndot, alphadot, vdot, Ddot, deltadot, thetadot)

    m = MX.sym("m")
    C1 = MX.sym("C1")
    C2 = MX.sym("C2")
    Cm1 = MX.sym("Cm1")
    Cm2 = MX.sym("Cm2")
    Cr0 = MX.sym("Cr0")
    Cr2 = MX.sym("Cr2")
    Cr3 = MX.sym("Cr3")
    Iz = MX.sym("Iz")
    lr = MX.sym("lr")
    lf = MX.sym("lf")
    Df = MX.sym("Df")
    Cf = MX.sym("Cf")
    Bf = MX.sym("Bf")
    Dr = MX.sym("Dr")
    Cr = MX.sym("Cr")
    Br = MX.sym("Br")
    Imax_c = MX.sym("Imax_c")
    Caccel = MX.sym("Caccel")
    qc = MX.sym("qc")
    ql = MX.sym("ql")
    gamma = MX.sym("gamma")
    r1 = MX.sym("r1")
    r2 = MX.sym("r2")
    r3 = MX.sym("r3")

    # algebraic variables
    z = vertcat([])

    # parameters
    p = vertcat(
        m,
        C1,
        C2,
        Cm1,
        Cm2,
        Cr0,
        Cr2,
        Cr3,
        Iz,
        lr,
        lf,
        Bf,
        Cf,
        Df,
        Br,
        Cr,
        Dr,
        Imax_c,
        Caccel,
        qc,
        ql,
        gamma,
        r1,
        r2,
        r3,
    )

    s_mod = mod(s, pathlength)

    beta = atan2(lr, lr + lf) * tan(next_delta)
    sdota = (v * cos(alpha + C1 * next_delta)) / (1 - kapparef_s(s_mod) * n)
    f_expl = vertcat(
        sdota,
        v * sin(alpha + C1 * next_delta),
        v / lr * sin(beta) - kapparef_s(s_mod) * sdota,
        # v * C2 * next_delta - kapparef_s(s_mod) * sdota,
        next_D * cos(C1 * next_delta),
        derD,
        derDelta,
        derTheta,
    )


    # constraint on forces
    a_lat = next_D * sin(C1 * next_delta)
    a_long = next_D

    n_outer_bound = outer_bound_s(s_mod) + n
    n_inner_bound = inner_bound_s(s_mod) - n

    # Model bounds
    model.n_min = -1e3
    model.n_max = 1e3

    constraint.n_min = cfg_dict["track_savety_margin"]  # width of the track [m]
    constraint.n_max = 1e3  # width of the track [m]
    # state bounds
    model.throttle_min = -5.0
    model.throttle_max = 5.0

    model.delta_min = -0.40  # minimum steering angle [rad]
    model.delta_max = 0.40  # maximum steering angle [rad]

    # input bounds
    model.ddelta_min = -1.0  # minimum change rate of stering angle [rad/s]
    model.ddelta_max = 1.0  # maximum change rate of steering angle [rad/s]
    model.dthrottle_min = -10  # -10.0  # minimum throttle change rate
    model.dthrottle_max = 10  # 10.0  # maximum throttle change rate
    model.dtheta_min = -3.2
    model.dtheta_max = 5

    # nonlinear constraint
    constraint.alat_min = -100  # maximum lateral force [m/s^2]
    constraint.alat_max = 100  # maximum lateral force [m/s^1]

    constraint.along_min = -4  # maximum lateral force [m/s^2]
    constraint.along_max = 4  # maximum lateral force [m/s^2]

    model.v_min = 0
    model.v_max = 30

    # Define initial conditions
    model.x0 = np.array([-2, 0, 0, 0, 0, 0, 0])

    model.cost_expr_ext_cost = (
        ql * (s - theta) ** 2
        + qc * n**2
        - gamma * derTheta
        + r1 * derD**2
        + r2 * derDelta**2
        + r3 * derTheta**2
    )
    model.cost_expr_ext_cost_e = 0
    # model.cost_expr_ext_cost_e = (
    #     ql * (s - theta) ** 2
    #     + qc * n**2
    # )

    # define constraints struct
    constraint.alat = Function("a_lat", [x, u], [a_lat])
    constraint.pathlength = pathlength
    constraint.expr = vertcat(a_long, a_lat, n_inner_bound, n_outer_bound)


    f_expl_func = Function(
        "f_expl_func", [s, n, alpha, v, D, delta, theta, derD, derDelta, derTheta, p], [f_expl]
    )


    # Define model struct
    params = types.SimpleNamespace()
    params.C1 = C1
    params.C2 = C2
    params.Cm1 = Cm1
    params.Cm2 = Cm2
    params.Cr0 = Cr0
    params.Cr2 = Cr2
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name
    model.params = params
    model.kappa = kapparef_s
    model.f_expl_func = f_expl_func
    model.outer_bound_s = outer_bound_s
    model.inner_bound_s = inner_bound_s
    return model, constraint, params
