#!/usr/bin/python3
from tkinter import Y
import rospy
import math
from enum import Enum
from dataclasses import dataclass

import tf2_ros
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from task1.srv import Normal_vector
from actionlib_msgs.msg import GoalStatusArray, GoalID
from geometry_msgs.msg import PoseStamped, Twist
from sound_play.libsoundplay import SoundClient
from task1.msg import FaceLocation


class State(Enum):
    IDLE = "idle"
    WANDERING = "wandering"
    ROTATING = "rotating"
    APPROACHING = "approaching"
    APPROACHING_POTENTIAL = "approaching potential"


@dataclass
class Goal_Pose():
    x: float
    y: float

    rot_z: float
    rot_w: float

    def to_pose_stamped(self):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.orientation.z = self.rot_z
        goal.pose.orientation.w = self.rot_w
        goal.pose.position.x = self.x
        goal.pose.position.y = self.y
        goal.header.stamp = rospy.Time.now()

        return goal

# TODO:
# add rotation

# add autonomous target creation sometime in the future


class MovementNode():
    def __init__(self):
        rospy.init_node('map_goals')

        self.positions = [
            Goal_Pose(0.0793, -0.5000, -0.9191, 0.3941),
            Goal_Pose(-0.0524, -1.1012, -0.6953, 0.7187),
            Goal_Pose(1.0350, -1.5649, 0.6870, 0.7267),
            Goal_Pose(2.8447, -0.8371, 0.0048, 1.0000),
            Goal_Pose(3.0377, -0.7304, 0.5307, 0.8476),
            Goal_Pose(1.2741, 0.0321, -0.8169, 0.5767),
            Goal_Pose(1.6957, 0.7032, -0.9999, 0.0119),
            Goal_Pose(1.6715, 0.7124, 0.1005, 0.9949),
            Goal_Pose(2.4370, 1.7626, 0.8433, 0.5375),
            Goal_Pose(1.3881, 2.3640, -0.8891, 0.4576),
            Goal_Pose(0.6501, 2.3621, 0.9404, 0.3401),
            Goal_Pose(-0.6384, 1.7437, 0.8824, 0.4706),
            Goal_Pose(-0.7544, 1.7158, -0.9659, 0.2589),
            Goal_Pose(-0.5732, 0.5604, 0.2789, 0.9603),
            Goal_Pose(-0.4047, 0.4713, -0.9752, 0.2213)
        ][::-1]

        self.goal_cnt = 0
        self.state = State.WANDERING
        # ignore old status messages sent before starting movement
        self.goal_sent = False
        self.status_sub = rospy.Subscriber(
            '/move_base/status', GoalStatusArray, self.status_callback)
        self.goal_pub = rospy.Publisher(
            '/move_base_simple/goal', PoseStamped, queue_size=1000)
        self.twist_pub = rospy.Publisher(
            '/cmd_vel_mux/input/teleop', Twist, queue_size=1000)
        self.goal_cancel = rospy.Publisher(
            "/move_base/cancel", GoalID, queue_size=1000)

        rospy.wait_for_service("normal_vector")
        self.vector_service = rospy.ServiceProxy(
            "normal_vector", Normal_vector)

        self.soundhandle = SoundClient()
        self.voice = 'voice_kal_diphone'
        #self.voice = 'voice_rab_diphone'
        #self.voice = 'voice_el_diphone'

        rospy.sleep(1)

        self.volume = 1.0
        self.soundhandle.say(
            "Hello world, nice to meet you!", self.voice, self.volume)
        self.met_faces = []

        # for position
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

        self.face_sub = rospy.Subscriber(
            '/task1/face', FaceLocation, self.face_callback)

        self.potential_face_sub = rospy.Subscriber(
            '/task1/potential_face', FaceLocation, self.potential_face_callback)

        self.do_movement()

        rate = rospy.Rate(2)
        while not rospy.is_shutdown():

            # Mogoce sploh ne bomo rabli tega rotationa
            if self.state is State.ROTATING:
                # self.rotate_once()
                # if rospy.get_time() - self.rotation_start > 1 and abs(self.get_heading() - self.start_heading) < 0.1:
                #     rospy.loginfo("Made full rotation")
                self.state = State.WANDERING
                self.do_movement()

            rate.sleep()

    def get_position(self):
        return self.tf2_buffer.lookup_transform(
            "map", "base_link", rospy.Time(0))

    def get_heading(self):
        pos = self.get_position()
        rot = pos.transform.rotation
        return euler_from_quaternion((rot.x, rot.y, rot.z, rot.w))[2]

    def stop_movement(self):
        self.goal_cancel.publish(GoalID())
        rospy.loginfo("Canceled previous goal")

    def do_movement(self):
        if self.goal_cnt >= len(self.positions):
            self.goal_cnt = 0
            rospy.loginfo("All goals reached, going again...")

        self.goal_pub.publish(self.positions[self.goal_cnt].to_pose_stamped())
        self.goal_sent = True
        rospy.loginfo(f"Published goal {self.goal_cnt}")

    def start_rotation(self):
        self.rotation_start = rospy.get_time()
        self.start_heading = self.get_heading()
        self.state = State.ROTATING

    def rotate_once(self):
        twist = Twist()
        twist.linear.x = twist.linear.y = twist.linear.z = 0
        twist.angular.x = twist.angular.y = 0
        twist.angular.z = math.radians(360/10)
        self.twist_pub.publish(twist)

    def status_callback(self, goalStatusArray):
        if self.state not in [State.WANDERING, State.APPROACHING_POTENTIAL, State.APPROACHING]:
            return

        if len(goalStatusArray.status_list) == 0 or not self.goal_sent:
            return

        goalStatus = goalStatusArray.status_list[-1]

        # Pending or active
        if goalStatus.status in [0, 1]:
            return

        if goalStatus.status == 3:
            if self.state == State.WANDERING:
                self.goal_cnt += 1
                self.start_rotation()

            elif self.state == State.APPROACHING_POTENTIAL:
                self.do_movement()
                self.state = State.WANDERING

            elif self.state == State.APPROACHING:
                self.greet_face()
                rospy.sleep(2)
                self.state = State.WANDERING
                self.do_movement()

            rospy.loginfo(goalStatus.text)

        # rospy.loginfo(goalStatus.text)
        # self.do_movement()

    def potential_face_callback(self, faceLocation):
        # come closer to see if there is a face
        # if faceLocation.id in self.met_faces:
        #     rospy.loginfo(f"Already greeted face {faceLocation.id}")
        #     return

        if self.state in [State.APPROACHING_POTENTIAL, State.APPROACHING]:
            return
        self.state = State.APPROACHING_POTENTIAL

        self.stop_movement()
        rospy.loginfo("Approaching a potential face")

        pos = self.get_position().transform.translation
        angle = self.vector_service(faceLocation.x, faceLocation.y).angle

        # mAGIC fUCKERY
        x_diff = math.cos(angle)
        y_diff = math.sin(-angle)
        yaw = angle

        if abs(y_diff) < 0.1:
            yaw += math.pi

        target_x = faceLocation.x - pos.x
        target_y = faceLocation.y - pos.y

        # Check if we are facing the normal
        if np.dot([x_diff, y_diff], [target_x, target_y]) > 0:
            x_diff *= -1
            y_diff *= -1
            yaw += math.pi

        quat = quaternion_from_euler(0, 0, yaw)

        # Set destination 0.5 units away from face
        final_x = faceLocation.x + x_diff * 0.5
        final_y = faceLocation.y + y_diff * 0.5

        target = Goal_Pose(
            x=final_x,
            y=final_y,
            rot_z=quat[2],
            rot_w=quat[3])

        self.goal_pub.publish(target.to_pose_stamped())
        self.goal_sent = True
        rospy.loginfo(
            f"Moving to {final_x}, {final_y}")

    def face_callback(self, faceLocation):
        # come closer to see if there is a face
        if faceLocation.id in self.met_faces:
            rospy.loginfo(f"Already greeted face {faceLocation.id}")
            return

        if self.state == State.APPROACHING:
            return

        self.current_face_id = faceLocation.id
        self.current_face_name = faceLocation.name
        self.met_faces.append(faceLocation.id)
        self.stop_movement()
        self.state = State.APPROACHING
        rospy.loginfo("Approaching a face")

        pos = self.get_position().transform.translation
        angle = self.vector_service(faceLocation.x, faceLocation.y).angle

        # mAGIC fUCKERY
        x_diff = math.cos(angle)
        y_diff = math.sin(-angle)
        yaw = angle

        if abs(y_diff) < 0.1:
            yaw += math.pi

        target_x = faceLocation.x - pos.x
        target_y = faceLocation.y - pos.y

        # Check if we are facing the normal
        if np.dot([x_diff, y_diff], [target_x, target_y]) > 0:
            x_diff *= -1
            y_diff *= -1
            yaw += math.pi

        quat = quaternion_from_euler(0, 0, yaw)

        # Set destination 0.5 units away from face
        final_x = faceLocation.x + x_diff * 0.5
        final_y = faceLocation.y + y_diff * 0.5

        target = Goal_Pose(
            x=final_x,
            y=final_y,
            rot_z=quat[2],
            rot_w=quat[3])

        self.goal_pub.publish(target.to_pose_stamped())
        self.goal_sent = True
        rospy.loginfo(
            f"Moving to {final_x}, {final_y}")

    def greet_face(self):
        rospy.loginfo(f"Greeted {len(self.met_faces)} faces")
        string_to_say = f'Hi, {self.current_face_name if self.current_face_name else f"face number {self.current_face_id}"}, nice to meet you!'
        self.soundhandle.say(string_to_say, self.voice, self.volume)


if __name__ == '__main__':
    MovementNode()
