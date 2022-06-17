#!/usr/bin/python3
import rospy
from geometry_msgs.msg import PoseStamped


class Listen():
    def __init__(self):

        rospy.init_node("listen")

        self.potential_face_sub = rospy.Subscriber(
            '/move_base_simple/goal', PoseStamped, self.goal_callback)

        rospy.spin()

    def goal_callback(self, goal):

        print(
            f"Goal_Pose({goal.pose.position.x:.4f}, {goal.pose.position.y:.4f}, {goal.pose.orientation.z:.4f}, {goal.pose.orientation.w:.4f}),")

        # print(goal)

        pass


def main():
    Listen()


if __name__ == '__main__':
    main()
