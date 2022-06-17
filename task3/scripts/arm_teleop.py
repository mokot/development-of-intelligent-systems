#!/usr/bin/env python3
import rospy

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import sys
import select
import termios
import tty

msg = """
Control Your Turtlebot arm!
---------------------------
Moving around:
    q w e r
    a s d f

CTRL-C to quit
"""

moveBindings = {
    'q': (0.1, 0, 0, 0),
    'w': (0, 0.1, 0, 0),
    'e': (0, 0, 0.1, 0),
    'r': (0, 0, 0, 0.1),
    'a': (-0.1, 0, 0, 0),
    's': (0, -0.1, 0, 0),
    'd': (0, 0, -0.1, 0),
    'f': (0, 0, 0, -0.1),
}


def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('arm_teleop')
    pub = rospy.Publisher('/turtlebot_arm/arm_controller/command', JointTrajectory, queue_size=1)

    angles = [0, -2, 2.6, -0.5]
    try:
        print(msg)

        import time
        prev = time.time()

        while(1):
            key = getKey()
            if key in moveBindings.keys():
                angles[0] += moveBindings[key][0]
                angles[1] += moveBindings[key][1]
                angles[2] += moveBindings[key][2]
                angles[3] += moveBindings[key][3]
                print(angles)

                trajectory = JointTrajectory()
                trajectory.joint_names = ["arm_shoulder_pan_joint",
                                          "arm_shoulder_lift_joint", "arm_elbow_flex_joint", "arm_wrist_flex_joint"]
                trajectory.points = [JointTrajectoryPoint(positions=angles, time_from_start=rospy.Duration(0.5))]
                pub.publish(trajectory)
                prev = time.time()
            else:
                if (key == '\x03'):
                    break

            #print("loop: {0}".format(count))
            #print("target: vx: {0}, wz: {1}".format(target_speed, target_turn))
            #print("publihsed: vx: {0}, wz: {1}".format(twist.linear.x, twist.angular.z))

    except Exception as e:
        print(e)

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
