#!/usr/bin/python3
import rospy
import cv2
import numpy as np
import random as rng
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node("prototyper")
bridge = CvBridge()

r = rospy.Rate(5)

while not rospy.is_shutdown():
    try:
        image_msg = rospy.wait_for_message("/arm_camera/rgb/image_raw", Image)
        image = bridge.imgmsg_to_cv2(image_msg, "mono8")
    except CvBridgeError as e:
        print(e)
        continue

    image = image[210:310, :]
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=50, maxRadius=400)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("image", image)
    cv2.waitKey(1)
    r.sleep()
