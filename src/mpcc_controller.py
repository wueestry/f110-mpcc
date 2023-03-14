#! /usr/bin/env python3

import os
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import rospy
import yaml
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import ObstacleArray, WpntArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from scipy.integrate import solve_ivp
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray

from mpc.acados_settings import acados_settings
from plotting_fnc import plot_res
from utils.frenet_cartesian_converter import convert_frenet_to_cartesian
from utils.indecies import Input, Parameter, State
from utils.splinify import SplineTrack


class MPC:
    def __init__(self, conf_file: str) -> None:
        rospy.init_node("mpc_node", anonymous=True, log_level=rospy.DEBUG)

        self.conf_file = conf_file
        self.vel_x = 0

        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self._odom_frenet_cb)
        rospy.Subscriber("/car_state/pose", PoseStamped, self._pose_cb)
        rospy.Subscriber(
            "/vesc/high_level/ackermann_cmd_mux/input/nav_1", AckermannDriveStamped, self._input_cb
        )
        rospy.Subscriber("/vesc/sensors/imu/raw", Imu, self._imu_cb)
        rospy.Subscriber("/obstacles", ObstacleArray, self._obstacle_cb)
        rospy.Subscriber("/tf_odom", Odometry, self._tf_odom_cb)

        self.pred_pos_pub = rospy.Publisher(
            "/mpc_controller/predicted_position", MarkerArray, queue_size=10
        )
        self.target_pos_pub = rospy.Publisher(
            "/mpc_controller/target_position", MarkerArray, queue_size=10
        )
        self.drive_pub = rospy.Publisher(
            "/mpc_controller/input_stream", AckermannDriveStamped, queue_size=10
        )
        self.drive_next_pub = rospy.Publisher(
            "/mpc_controller/next_input", AckermannDriveStamped, queue_size=10
        )
        self.d_left_pub = rospy.Publisher("/mpc_controller/d_left", MarkerArray, queue_size=10)
        self.d_mid_pub = rospy.Publisher("/mpc_controller/d_mid", MarkerArray, queue_size=10)
        self.d_right_pub = rospy.Publisher("/mpc_controller/d_right", MarkerArray, queue_size=10)
        self.d_left_adj_pub = rospy.Publisher(
            "/mpc_controller/d_left_adj", MarkerArray, queue_size=10
        )
        self.d_right_adj_pub = rospy.Publisher(
            "/mpc_controller/d_right_adj", MarkerArray, queue_size=10
        )
        self.pos_pub = rospy.Publisher("/mpc_controller/current_position", Marker, queue_size=5)
        self.pos_n_pub = rospy.Publisher("/mpc_controller/current_pos_n", Odometry, queue_size=10)

    def initialize(self) -> None:
        try:
            shortest_path = rospy.wait_for_message("/global_waypoints", WpntArray, 20.0)
        except:
            raise TimeoutError("No waypoints received in the appropriate amount of time.")

        # Used for wrapping
        s_coords = [x.s_m for x in shortest_path.wpnts]

        d_left, coords_path, d_right = self._transform_waypoints_to_coords(
            shortest_path.wpnts
        )  # on f track trajectory is 81.803 m long.

        self.spline = SplineTrack(coords_direct=coords_path)

        with open(self.conf_file, "r") as file:
            cfg = yaml.safe_load(file)
            for key in cfg.keys():
                if type(cfg[key]) is list:
                    cfg[key] = [float(i) for i in cfg[key]]

        self.Tf = cfg["Tf"]
        self.N = cfg["N"]
        self.T = cfg["T"]
        self.sref_N = cfg["sref_N"]
        self.s_offset = cfg["s_offset"]
        self.track_savety_margin = cfg["track_savety_margin"]
        self.slip_angle_approx = cfg["slip_angle_approximation"]
        self.use_pacejka = cfg["use_pacejka_tiremodel"]
        t_delay = cfg["t_delay"]
        t_MPC = 1 / cfg["MPC_freq"]

        # time delay propagation
        self.t_delay = t_delay + t_MPC
        self.Ts = t_MPC

        self.nr_laps = 0

        kapparef = [x.kappa_radpm for x in shortest_path.wpnts]
        s0 = self.spline.params

        self.constraint, self.model, self.acados_solver, self.model_params = acados_settings(
            self.Ts, self.N, s0, kapparef, d_left, d_right, cfg
        )

        self.obstacles = None

        self.kappa = self.model.kappa

    def _odom_cb(self, data: Odometry) -> None:
        self.vel_x = data.twist.twist.linear.x
        self.omega = data.twist.twist.angular.z

    def _odom_frenet_cb(self, data: Odometry) -> None:
        self.pos_s = data.pose.pose.position.x
        self.pos_n = data.pose.pose.position.y

    def _pose_cb(self, data: PoseStamped) -> None:
        self.pos_x = data.pose.position.x
        self.pos_y = data.pose.position.y

        self.theta = euler_from_quaternion(
            [
                data.pose.orientation.x,
                data.pose.orientation.y,
                data.pose.orientation.z,
                data.pose.orientation.w,
            ]
        )[2]

    def _input_cb(self, data: AckermannDriveStamped) -> None:
        self.steering_angle = data.drive.steering_angle

    def _imu_cb(self, data: Imu) -> None:
        self.acceleration = data.linear_acceleration.x  # Checked it with plotting. Should be fine.

    def _obstacle_cb(self, data: ObstacleArray) -> None:
        self.obstacles = data.obstacles

    def _tf_odom_cb(self, data: Odometry) -> None:
        self.vel_y = data.twist.twist.linear.y

    def _transform_waypoints_to_coords(
        self, data: WpntArray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        waypoints = np.zeros((len(data), 2))
        d_left = np.zeros(len(data))
        d_right = np.zeros(len(data))
        boundaries = np.zeros((len(data), 2))
        for idx, wpnt in enumerate(data):
            waypoints[idx] = [wpnt.x_m, wpnt.y_m]
            d_left[idx] = wpnt.d_right  # Fix for boundaries
            d_right[idx] = wpnt.d_left  # Fix for boundaries
        res_coords = np.array([boundaries[:-1], waypoints[:-1], boundaries[:-1]])
        return d_left, res_coords, d_right

    def _dynamics_of_car(self, t, x0) -> list:
        """
        Used for forward propagation. This function takes the dynamics from the acados model.
        """
        s = x0[State.POS_ON_CENTER_LINE_S]
        n = x0[State.MIN_DIST_TO_CENTER_LINE_N]
        alpha = x0[State.ORIENTATION_ALPHA]
        vx = max(0.1, x0[State.VELOCITY_VX])
        vy = x0[State.VELOCITY_VY]
        omega = x0[State.YAW_RATE_OMEGA]
        D = x0[State.DUTY_CYCLE_D]
        delta = x0[State.STEERING_ANGLE_DELTA]
        theta = x0[State.PROGRESS_THETA]

        derD = x0[len(State) + Input.D_DUTY_CYCLE]
        derDelta = x0[len(State) + Input.D_STEERING_ANGLE]
        derTheta = x0[len(State) + Input.D_PROGRESS]

        m = self.model_params.p[Parameter.m]
        Imax_c = self.model_params.p[Parameter.Imax_c]
        Cr0 = self.model_params.p[Parameter.Cr0]
        Caccel = self.model_params.p[Parameter.Caccel]
        Cdecel = self.model_params.p[Parameter.Cdecel]
        lr = self.model_params.p[Parameter.lr]
        lf = self.model_params.p[Parameter.lf]
        CSr = self.model_params.p[Parameter.CSr]
        CSf = self.model_params.p[Parameter.CSf]
        Dr = self.model_params.p[Parameter.Dr]
        Df = self.model_params.p[Parameter.Df]
        Cr = self.model_params.p[Parameter.Cr]
        Cf = self.model_params.p[Parameter.Cf]
        Br = self.model_params.p[Parameter.Br]
        Bf = self.model_params.p[Parameter.Bf]
        Iz = self.model_params.p[Parameter.Iz]

        def accel(vx: float, D: float) -> float:
            return m * (Imax_c - Cr0 * vx) * D / (self.model.throttle_max * Caccel)

        def decel(vx: float, D: float) -> float:
            return m * (-Imax_c - Cr0 * vx) * abs(D) / (self.model.throttle_max * Cdecel)

        Fx = accel(vx, D) if D >= 0 else decel(vx, D)

        if self.slip_angle_approx:
            beta = np.arctan2(vy, vx)
            ar = -beta + lr * omega / vx
            af = delta - beta - lf * omega / vx
        else:
            af = -np.arctan2(vy + lf * omega, vx) + delta
            ar = -np.arctan2(vy - lr * omega, vx)

        Fr = CSr * ar
        Ff = CSf * af

        if self.use_pacejka:
            Fr = Dr * np.sin(Cr * np.arctan(Br * ar))
            Ff = Df * np.sin(Cf * np.arctan(Bf * af))

        xdot = [
            (vx * np.cos(alpha) - vy * np.sin(alpha)) / (1 - float(self.model.kappa(s)) * n),
            vx * np.sin(alpha) + vy * np.cos(alpha),
            omega,
            1 / m * (Fx - Ff * np.sin(delta) + m * vy * omega),
            1 / m * (Fr + Ff * np.cos(delta) - m * vx * omega),
            1 / Iz * (Ff * lf * np.cos(delta) - Fr * lr),
            derD,
            derDelta,
            derTheta,
            derD,
            derDelta,
            derTheta,
        ]

        return xdot

    def propagate_time_delay(self, states: np.array, inputs: np.array) -> np.array:

        # Initial condition on the ODE
        x0 = np.concatenate((states, inputs), axis=0)

        solution = solve_ivp(
            self._dynamics_of_car,
            t_span=[0, self.t_delay],
            y0=x0,
            method="RK45",
            atol=1e-8,
            rtol=1e-8,
        )

        solution = [x[-1] for x in solution.y]

        # Constraint on max. steering angle
        if abs(solution[State.STEERING_ANGLE_DELTA]) > self.model.delta_max:
            solution[State.STEERING_ANGLE_DELTA] = (
                np.sign(solution[State.STEERING_ANGLE_DELTA]) * self.model.delta_max
            )

        # Constraint on max. thrust
        if abs(solution[State.DUTY_CYCLE_D]) > self.model.throttle_max:
            solution[State.DUTY_CYCLE_D] = (
                np.sign(solution[State.DUTY_CYCLE_D]) * self.model.throttle_max
            )

        # Only get the state as solution of where the car will be in t_delay seconds
        return np.array(solution)[: -len(Input)]

    def control_loop(self) -> None:
        self.initialize()
        rate = rospy.Rate(1 / self.Ts)

        x0 = self.get_initial_position(init=True)
        propagated_x = self.propagate_time_delay(x0, np.zeros(len(Input)))

        self.pred_traj = np.array([self.s_offset + x0[State.POS_ON_CENTER_LINE_S] + self.sref_N * j / self.N for j in range(self.N)])

        self.acados_solver.set(0, "lbx", propagated_x)
        self.acados_solver.set(0, "ubx", propagated_x)

        self.lap_times = [time.perf_counter()]
        self.nr_of_failures = 0
        self.qp_iterations = []

        simX = np.ndarray((int(1e5), self.model.x.size()[0]))
        simU = np.ndarray((int(1e5), self.model.u.size()[0]))
        realX = np.ndarray((int(1e5), self.model.x.size()[0]))
        propX = np.ndarray((int(1e5), self.model.x.size()[0]))
        iter_loop = 1
        simX[0, :] = x0
        realX[0, :] = x0
        propX[0, :] = propagated_x

        tcomp_sum = 0
        tcomp_max = 0

        while not rospy.is_shutdown():

            start = time.perf_counter()
            self.lb_list = np.ones(self.N - 1) * (-1e3)
            self.ub_list = np.ones(self.N - 1) * (1e3)
            self.traj_list = self.pred_traj[1:]
            if self.obstacles is not None:
                for j in range(1, self.N):
                    s_traj_mod = self.pred_traj[j] % self.spline.track_length
                    traj_ub = self.model.inner_bound_s(s_traj_mod) - self.track_savety_margin
                    traj_lb = -self.model.outer_bound_s(s_traj_mod) + self.track_savety_margin
                    for obstacle in self.obstacles:
                        obs_right_shifted = obstacle.d_right - traj_lb
                        obs_left_shifted = obstacle.d_left - traj_lb
                        if (
                            s_traj_mod >= obstacle.s_start - 0.5
                            and s_traj_mod <= obstacle.s_end + 0.1
                        ):
                            gap_left = traj_ub - obs_left_shifted - traj_lb
                            gap_right = obs_right_shifted
                            if gap_right >= gap_left:
                                traj_ub = (
                                    obstacle.d_right - 0.25
                                    if obstacle.d_right < traj_ub
                                    else traj_ub
                                )
                            else:
                                traj_lb = (
                                    obstacle.d_left + 0.25
                                    if obstacle.d_left > traj_lb
                                    else traj_lb
                                )

                    self.lb_list[j - 1] = traj_lb
                    self.ub_list[j - 1] = traj_ub

                    lbx = np.array(
                        [
                            traj_lb,
                            self.model.v_min,
                            self.model.throttle_min,
                            self.model.delta_min,
                        ]
                    )

                    ubx = np.array(
                        [
                            traj_ub,
                            self.model.v_max,
                            self.model.throttle_max,
                            self.model.delta_max,
                        ]
                    )

                    self.acados_solver.set(j, "lbx", lbx)
                    self.acados_solver.set(j, "ubx", ubx)

            status = self.acados_solver.solve()

            if status != 0:
                rospy.logerr(f"acados returned status {status} in closed loop iteration.")
                self.nr_of_failures += 1

            # get solution
            x0 = self.acados_solver.get(0, "x")
            self.u0 = self.acados_solver.get(0, "u")
            self.pred_x = self.acados_solver.get(1, "x")

            if status == 0:
                self.publish_ackermann_msg(self.pred_x, True)

                for stage in range(5, 1, -1):
                    x = self.acados_solver.get(stage, "x")
                    self.publish_ackermann_msg(x, False)

            simX[iter_loop, :] = self.pred_x
            simU[iter_loop, :] = self.u0

            print(iter_loop)

            # Creating waypoint array with predicted positions
            mpc_sd = np.array([self.acados_solver.get(j, "x")[:2] for j in range(self.N)])

            pred_waypoints = convert_frenet_to_cartesian(self.spline, mpc_sd)

            self.pred_traj = np.array(
                [self.acados_solver.get(j, "x")[State.PROGRESS_THETA] for j in range(self.N)]
            )

            progress_waypoints = convert_frenet_to_cartesian(self.spline, np.array([self.pred_traj,np.zeros(len(self.pred_traj))]).T)

            self.publish_waypoint_markers(pred_waypoints, type="pred")
            self.publish_waypoint_markers(progress_waypoints, type="target")

            # self.publish_current_pos_n(x0)

            D0 = x0[State.DUTY_CYCLE_D]
            delta0 = x0[State.STEERING_ANGLE_DELTA]

            x0 = self.get_initial_position(
                init=False, prev_x0=x0
            )  # Take D and delta from MPC solution!

            x0[State.DUTY_CYCLE_D] = D0
            x0[State.STEERING_ANGLE_DELTA] = delta0

            self.qp_iterations.append(sum(self.acados_solver.get_stats("qp_iter")))

            propagated_x = self.propagate_time_delay(x0, self.u0)
            prop_x_plot = convert_frenet_to_cartesian(self.spline, propagated_x[:2])
            self.publish_current_pos(prop_x_plot)

            # (f"s: {x0[0]}, n: {x0[1]}, alpha: {x0[2]}, v: {x0[3]}, D: {x0[4]}, delta: {x0[5]}")

            realX[iter_loop, :] = x0
            propX[iter_loop, :] = propagated_x

            # Get current position of the car which is taken as x0 for the next interation
            # self.publish_current_pos(self.spline.get_coordinate(x0[0])+x0[1]*self.spline.get_derivative(x0[0]) @ R)

            self.acados_solver.set(0, "lbx", propagated_x)
            self.acados_solver.set(0, "ubx", propagated_x)

            # Print trajectory bounds
            boundaries = np.zeros((self.acados_solver.N + 1, 3, 2))
            boundaries_adj = np.zeros((self.N - 1, 2, 2))

            for stage in range(self.acados_solver.N + 1):
                x_ = self.acados_solver.get(stage, "x")
                s_ = x_[State.POS_ON_CENTER_LINE_S]
                s_mod = s_ % self.spline.track_length
                n_ = x_[State.MIN_DIST_TO_CENTER_LINE_N]
                if stage == 1:
                    if (
                        self.model.outer_bound_s(s_mod) + n_ <= 0.4
                        or self.model.inner_bound_s(s_mod) - n_ <= 0.4
                    ):
                        rospy.logwarn(
                            f"Outer: {self.model.outer_bound_s(s_mod) + n_}, Inner: {self.model.inner_bound_s(s_mod) - n_}"
                        )

                boundaries[stage, 0, :] = convert_frenet_to_cartesian(
                    self.spline,
                    np.array([s_mod, -self.model.outer_bound_s(s_mod) + self.track_savety_margin]),
                )
                boundaries[stage, 2, :] = convert_frenet_to_cartesian(
                    self.spline,
                    np.array([s_mod, self.model.inner_bound_s(s_mod) - self.track_savety_margin]),
                )
                boundaries[stage, 1, :] = convert_frenet_to_cartesian(
                    self.spline, np.array([s_mod, 0])
                )

                if stage < self.N - 1:
                    x_traj = self.traj_list[stage]
                    boundaries_adj[stage, 0, :] = convert_frenet_to_cartesian(
                        self.spline, np.array([x_traj, self.lb_list[stage]])
                    )
                    boundaries_adj[stage, 1, :] = convert_frenet_to_cartesian(
                        self.spline, np.array([x_traj, self.ub_list[stage]])
                    )

            self.publish_waypoint_markers(boundaries[:, 0, :], "d_left")
            self.publish_waypoint_markers(boundaries[:, 1, :], "d_mid")
            self.publish_waypoint_markers(boundaries[:, 2, :], "d_right")
            self.publish_waypoint_markers(boundaries_adj[:, 0, :], "d_left_adj")
            self.publish_waypoint_markers(boundaries_adj[:, 1, :], "d_right_adj")

            iter_loop += 1

            elapsed = time.perf_counter() - start
            # manage timings
            tcomp_sum += elapsed
            if elapsed > tcomp_max:
                tcomp_max = elapsed

            # print(f"MPC took {elapsed:01.5f} s.")

            if iter_loop == 5000:
                save_plot = True
                simX_plot = simX[:iter_loop, :]
                simU_plot = simU[:iter_loop, :]
                realX_plot = realX[:iter_loop, :]
                print(f"Max. computation time: {tcomp_max}")
                print(f"Average computation time: {tcomp_sum/iter_loop}")
                print("Average speed:{}m/s".format(np.average(simX_plot[:, 3])))
                print(f"Lap times: {str(self.lap_times)[:-1]}")

                if save_plot:
                    filename = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
                    print(f"Saving files with name: {filename}")
                    with open(f"{filename}_info.txt", "w") as f:
                        f.write(f"Max. computation time: {tcomp_max} \n")
                        f.write(f"Average computation time: {tcomp_sum/iter_loop} \n")
                        f.write("Average speed:{}m/s \n".format(np.average(simX_plot[:, 3])))
                        f.write(f"Lap times: {str(self.lap_times)[:-1]} \n")
                        f.write(f"Nr of failures: {str(self.nr_of_failures)} \n")
                        f.write(
                            f"Average nr of iterations required: {str(np.mean(self.qp_iterations))} \n"
                        )

                plot_res(self.spline, simX_plot, simU_plot, realX_plot, save_plot, filename)

            rate.sleep()

    def get_initial_position(self, init: bool = False, prev_x0=None) -> np.ndarray:
        if init:
            self.pred_x = np.zeros(6)
            self.steering_angle = 0
            self.acceleration = 0
            prev_x0 = self.pred_x

        deriv = self.spline.get_derivative(self.pos_s)
        alpha = self.theta - np.arctan2(deriv[1], deriv[0])

        alpha = alpha % (2 * np.pi)
        if alpha > np.pi:
            alpha = alpha - 2 * np.pi

        # print(f"alpha: {alpha}")

        duty_cicle = (
            self.acceleration
        )  # Should be fine in simulation. On the car: -y coordinate of IMU
        delta = (
            self.steering_angle
        )  # ToDo: check if this is correct -> minus needed since steering_angle has been inverted

        track_length = self.spline.track_length - 0.1  # Needed as values are not exact

        if (
            self.pos_s < 0.2
            and prev_x0[State.POS_ON_CENTER_LINE_S] // track_length != self.nr_laps
        ):
            rospy.logdebug(
                f"---------------------------------LAP {int(self.nr_laps)} FINISHED------------------------------------"
            )
            self.lap_times[-1] = time.perf_counter() - self.lap_times[-1]
            self.nr_laps = prev_x0[State.POS_ON_CENTER_LINE_S] // track_length
            self.lap_times.append(time.perf_counter())

        current_pos_s = self.pos_s + self.nr_laps * self.spline.track_length

        return np.array(
            [
                current_pos_s,
                self.pos_n,
                alpha,
                self.vel_x,
                self.vel_y,
                duty_cicle,
                self.omega,
                delta,
                current_pos_s,
            ]
        )

    def publish_current_pos(self, coord: np.array) -> None:
        waypoint_marker = Marker()
        waypoint_marker.header.frame_id = "map"
        waypoint_marker.header.stamp = rospy.Time.now()
        waypoint_marker.type = 2
        waypoint_marker.scale.x = 0.2
        waypoint_marker.scale.y = 0.2
        waypoint_marker.scale.z = 0.2
        waypoint_marker.color.r = 0.0
        waypoint_marker.color.g = 1.0
        waypoint_marker.color.b = 0.0
        waypoint_marker.color.a = 1.0
        waypoint_marker.pose.position.x = coord[0]
        waypoint_marker.pose.position.y = coord[1]
        waypoint_marker.pose.position.z = 0
        waypoint_marker.pose.orientation.x = 0
        waypoint_marker.pose.orientation.y = 0
        waypoint_marker.pose.orientation.z = 0
        waypoint_marker.pose.orientation.w = 1
        waypoint_marker.id = 1
        self.pos_pub.publish(waypoint_marker)

    def publish_waypoint_markers(self, waypoints: np.ndarray, type: str) -> None:
        # rospy.logdebug("Publish waypoints")
        waypoint_markers = MarkerArray()
        wpnt_id = 0

        for waypoint in waypoints:
            waypoint_marker = Marker()
            waypoint_marker.header.frame_id = "map"
            waypoint_marker.header.stamp = rospy.Time.now()
            waypoint_marker.type = 2
            waypoint_marker.scale.x = 0.1
            waypoint_marker.scale.y = 0.1
            waypoint_marker.scale.z = 0.1
            if type == "pred":
                waypoint_marker.color.r = 1.0
                waypoint_marker.color.g = 0.0
                waypoint_marker.color.b = 1.0
                waypoint_marker.color.a = 1.0
            elif type == "target":
                waypoint_marker.color.r = 1.0
                waypoint_marker.color.g = 1.0
                waypoint_marker.color.b = 0.0
                waypoint_marker.color.a = 1.0
            else:
                waypoint_marker.color.r = 1.0
                waypoint_marker.color.g = 0.0
                waypoint_marker.color.b = 0.0
                waypoint_marker.color.a = 1.0

            waypoint_marker.pose.position.x = waypoint[0]
            waypoint_marker.pose.position.y = waypoint[1]
            waypoint_marker.pose.position.z = 0
            waypoint_marker.pose.orientation.x = 0
            waypoint_marker.pose.orientation.y = 0
            waypoint_marker.pose.orientation.z = 0
            waypoint_marker.pose.orientation.w = 1
            waypoint_marker.id = wpnt_id + 1
            wpnt_id += 1
            waypoint_markers.markers.append(waypoint_marker)

        if type == "pred":
            self.pred_pos_pub.publish(waypoint_markers)
        elif type == "d_left":
            self.d_left_pub.publish(waypoint_markers)
        elif type == "d_mid":
            self.d_mid_pub.publish(waypoint_markers)
        elif type == "d_right":
            self.d_right_pub.publish(waypoint_markers)
        elif type == "d_left_adj":
            self.d_left_adj_pub.publish(waypoint_markers)
        elif type == "d_right_adj":
            self.d_right_adj_pub.publish(waypoint_markers)
        else:
            self.target_pos_pub.publish(waypoint_markers)

    def publish_current_pos_n(self, state: np.array) -> None:
        position = Odometry()
        position.header.stamp = rospy.Time.now()
        position.header.frame_id = "base_link"
        position.pose.pose.position.y = state[State.MIN_DIST_TO_CENTER_LINE_N]
        self.pos_n_pub.publish(position)

    def publish_ackermann_msg(self, state: np.ndarray, next_input: bool = True) -> None:
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = rospy.Time.now()
        ack_msg.header.frame_id = "base_link"

        ack_msg.drive.steering_angle = state[State.STEERING_ANGLE_DELTA]
        # ack_msg.drive.steering_angle = -self.u0[-1]*0.16
        ack_msg.drive.speed = state[State.VELOCITY_VX]
        # print(f"Commanded steering angle: {ack_msg.drive.steering_angle}")

        # rospy.logdebug(f"Publish ackermann msg: {ack_msg.drive.speed}, {ack_msg.drive.steering_angle}")

        if next_input:
            self.drive_next_pub.publish(ack_msg)
        else:
            self.drive_pub.publish(ack_msg)


if __name__ == "__main__":
    dir_path = os.path.dirname(__file__)
    file_path = "mpc/param_config.yaml"
    controller = MPC(conf_file=os.path.join(dir_path, file_path))
    controller.control_loop()
