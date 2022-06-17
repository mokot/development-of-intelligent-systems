#!/usr/bin/env python3
import rospy

from enum import Enum
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String


class State(Enum):
    IDLE = 0
    ROTATING = 1


class ArmController:
    def __init__(self):
        rospy.init_node("arm_controller")
        self.pub = rospy.Publisher(
            "/turtlebot_arm/arm_controller/command", JointTrajectory, queue_size=1
        )

        rospy.Subscriber("/task3/arm_controller/command", String, self.command_callback)

        self.state = State.IDLE

        self.scanner_angles = [0, -2, 2.6, -0.5]
        self.parking_angles = [0, -0.8, 1.9, 0.1]
        self.self_destruct = [0, -3, 0, 0]
        self.direction = True
        self.max_angle = 3.14

        self.scan_duration = 2

        r = rospy.Rate(1 / self.scan_duration)
        while not rospy.is_shutdown():
            if self.state == State.ROTATING:
                self.scanner_angles[0] = (
                    self.max_angle if self.direction else -self.max_angle
                )
                self.set_position(self.scanner_angles, duration=self.scan_duration)
                self.direction = not self.direction
            r.sleep()

    def set_position(self, angles, duration=1):
        trajectory = JointTrajectory()
        trajectory.joint_names = [
            "arm_shoulder_pan_joint",
            "arm_shoulder_lift_joint",
            "arm_elbow_flex_joint",
            "arm_wrist_flex_joint",
        ]
        trajectory.points = [
            JointTrajectoryPoint(
                positions=angles, time_from_start=rospy.Duration(duration)
            )
        ]
        self.pub.publish(trajectory)

    def command_callback(self, msg):
        print("TUKAJ")
        rospy.loginfo("Received command: %s", msg.data)
        if msg.data == "setup_scanner":
            self.scanner_angles = [0, -2, 2.6, -0.5]
            self.setup_scanner_pos()
        if msg.data == "still_scanner":
            self.scanner_angles = [0, -2, 2.6, -0.5]
            self.set_position(self.scanner_angles)
            self.state = State.IDLE
        elif msg.data == "park":
            self.set_position(self.parking_angles)
            self.state = State.IDLE
        elif msg.data == "self_destruct":
            self.set_position(self.self_destruct)
            self.state = State.IDLE
        elif msg.data == "move_left":
            self.scanner_angles[0] = self.max_angle / 2
            self.set_position(self.scanner_angles)
            self.state = State.IDLE
        elif msg.data == "move_right":
            self.scanner_angles[0] = -self.max_angle / 2
            self.set_position(self.scanner_angles)
            self.state = State.IDLE
        elif msg.data == "food_detection":
            self.scanner_angles = [0, 0.3, 1, -0.5]
            self.set_position(self.scanner_angles)
            self.state = State.IDLE

    def setup_scanner_pos(self):
        self.set_position(self.scanner_angles)
        self.state = State.ROTATING


if __name__ == "__main__":
    ArmController()
