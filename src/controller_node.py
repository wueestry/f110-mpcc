#! /usr/bin/env python3

import os

import rospy
import yaml
from ackermann_msgs.msg import AckermannDriveStamped


class MPCController:
    def __init__(self, conf_file) -> None:
        with open(conf_file, "r") as file:
            cfg = yaml.safe_load(file)
            for key in cfg.keys():
                if type(cfg[key]) is list:
                    cfg[key] = [float(i) for i in cfg[key]]

        self.Hz = cfg["MPC_freq"]
        rospy.init_node("controller_node", anonymous=True, log_level=rospy.DEBUG)

        self.input_stream = []
        self.next_input = []

        rospy.Subscriber(
            "/mpc_controller/input_stream", AckermannDriveStamped, self._input_stream_cb
        )
        rospy.Subscriber("/mpc_controller/next_input", AckermannDriveStamped, self._next_input_cb)
        self.drive_pub = rospy.Publisher(
            "/vesc/high_level/ackermann_cmd_mux/input/nav_1", AckermannDriveStamped, queue_size=10
        )

    def _input_stream_cb(self, data: AckermannDriveStamped) -> None:
        self.input_stream.append(data)

        if len(self.input_stream) >= 50:
            self.input_stream = self.input_stream[-10:]  # Remove old values

    def _next_input_cb(self, data: AckermannDriveStamped) -> None:
        self.next_input.append(data)

    def send_ackermann_cmd(self) -> None:
        if len(self.next_input) == 0:
            rospy.logwarn(f"No new solution found. Taking previous value!")
            ack_msg = self.input_stream.pop()
        else:
            ack_msg = self.next_input.pop()

        # ack_msg.header.stamp = rospy.Time.now()

        rospy.logdebug(
            f"Publish ackermann msg: {ack_msg.drive.speed}, {ack_msg.drive.steering_angle}"
        )

        self.drive_pub.publish(ack_msg)

    def loop(self) -> None:
        rate = rospy.Rate(self.Hz)

        while not rospy.is_shutdown():

            if len(self.input_stream) == 0 and len(self.next_input) == 0:
                continue
                raise Exception("Input stream is currently empty")

            self.send_ackermann_cmd()

            rate.sleep()


if __name__ == "__main__":
    dir_path = os.path.dirname(__file__)
    file_path = "mpc/param_config.yaml"
    controller = MPCController(conf_file=os.path.join(dir_path, file_path))
    controller.loop()
