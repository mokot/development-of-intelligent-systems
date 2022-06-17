#!/usr/bin/env python3
import rospy

from task1.srv import Normal_vector, Normal_vectorResponse
from nav_msgs.msg import OccupancyGrid
import os
import cv2
import numpy as np
import math

position = None
resolution = None
normal_image = None
wall_image = None


def get_normal(req):
    pixel_x = int((req.x - position.x) / resolution)
    pixel_y = normal_image.shape[1] - int((req.y - position.y) / resolution)

    if wall_image[pixel_y, pixel_x] != 0:
        closest_dist = 99999
        wall_x = -1
        wall_y = -1

        N = 10
        for i in range(-N, N+1):
            for j in range(-N, N+1):
                if wall_image[pixel_y+i, pixel_x+j] == 0:
                    dist = i*i + j*j
                    if dist < closest_dist:
                        closest_dist = dist
                        wall_x = pixel_x + j
                        wall_y = pixel_y + i

        pixel_x = wall_x
        pixel_y = wall_y

    angle = np.median(normal_image[pixel_y-1:pixel_y+2, pixel_x-1:pixel_x+2])

    rospy.loginfo(f"Angle at {pixel_x},{pixel_y} is {angle}")
    return Normal_vectorResponse(angle)


def load_map(_map):
    global position
    global resolution
    position = _map.info.origin.position
    resolution = _map.info.resolution


def main():
    rospy.init_node("normal_vector_service")
    rospy.Subscriber("map", OccupancyGrid, load_map)

    rospy.Service("normal_vector", Normal_vector, get_normal)

    directory = os.path.dirname(__file__)
    global normal_image
    global wall_image

    normal_image = cv2.imread(os.path.join(
        directory, "../map/normal_map.png"), cv2.IMREAD_GRAYSCALE)
    normal_image = np.interp(
        normal_image, (normal_image.min(), normal_image.max()), (-math.pi, math.pi))

    wall_image = cv2.imread(os.path.join(
        directory, "../map/edited_map.pgm"), cv2.IMREAD_GRAYSCALE)

    rospy.spin()


if __name__ == "__main__":
    main()
