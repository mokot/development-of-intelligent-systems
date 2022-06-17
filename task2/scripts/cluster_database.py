#!/usr/bin/python3

import math
import rospy
import numpy as np
from geometry_msgs.msg import Vector3, Pose, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
import numpy as np

# Keep this
import tf2_geometry_msgs

from task2.msg import ColorPose
import webcolors


def rgba(r, g, b):
    return ColorRGBA(r / 255, g / 255, b / 255, 1)


def closest_colour(requested_colour):
    min_colours = {}

    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():

        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2

        min_colours[(rd + gd + bd)] = name

    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):

    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None

    return actual_name, closest_name


class ClusterObject:
    def __init__(self, pose: ColorPose, id: int) -> None:
        self.pose: Pose = pose.pose

        self.colors: list[ColorRGBA] = [pose.color]

        self.id: int = id
        self.update_counter: int = 1
        self.emmited: bool = False

        self.named_color: str = ""
        self.picked_color: ColorRGBA = rgba(0, 0, 0)

    def define_hsv(self):

        named_c = [get_colour_name((c.r, c.g, c.b))[1] for c in self.colors]

        # get the most common color
        most_common = max(set(named_c), key=named_c.count)

        # get the index of the most common color
        most_common_index = named_c.index(most_common)

        self.picked_color = self.colors[most_common_index]

        self.named_color = most_common

        rospy.loginfo("Color is {}".format(self.named_color))

    def update_position(self, update_pose: Pose, update_color: ColorRGBA) -> Pose:
        c_p: Point = self.pose.position
        u_p: Point = update_pose.position
        w: int = self.update_counter

        self.colors.append(update_color)

        self.pose = Pose(position=Point(x=(c_p.x * w + u_p.x) / (w+1),
                                        y=(c_p.y * w + u_p.y) / (w+1),
                                        z=(c_p.z * w + u_p.z) / (w+1)))

        self.update_counter += 1

        if self.emmited == False and self.update_counter > 3:
            self.define_hsv()
            self.emmited = True

        return self.pose

    def get_distance_to_pose(self, pose: Pose) -> float:
        c_p = self.pose.position
        o_p = pose.position

        return math.sqrt((c_p.x - o_p.x)**2 + (c_p.y - o_p.y)**2)


class CluserDatabase:
    def __init__(self, subscribe_topics: "list[str]", marker_topic: str, number_of_objects: int,  threshold: float = 0.2, ) -> None:

        for topic in subscribe_topics:
            rospy.Subscriber(topic, ColorPose, self.callback)

        self.objects: list[ClusterObject] = []
        self.threshold: float = threshold

        self.number_of_objects_to_detect = number_of_objects

        self.markers_pub: rospy.Publisher = rospy.Publisher(
            marker_topic, MarkerArray, queue_size=1000)

        self.task_pub: rospy.Publisher = rospy.Publisher(
            'task2/all_circles', Pose, queue_size=1000)

        # Log ros info to console with parameters
        rospy.loginfo("Starting cluster_database node with parameters:")
        rospy.loginfo("Subscribing to topics: %s", ", ".join(subscribe_topics))
        rospy.loginfo("Publishing to topic: %s", marker_topic)
        rospy.loginfo("Threshold: %s", threshold)

        self.publshed = False

        self.draw_markers()

    def callback(self, pose: ColorPose):
        self.add_object(pose)
        self.draw_markers()

        self.check_emmited()

    def check_emmited(self):

        if (self.publshed):
            return

        number_of_obj = [x.emmited for x in self.objects].count(True)

        if (number_of_obj == self.number_of_objects_to_detect):

            green_objects = []

            for obj in self.objects:
                name = obj.named_color.lower()

                if "green" in name or "lime" in name or "chartreuse" in name:
                    green_objects.append(obj)

            self.task_pub.publish(green_objects[0].pose)

            self.publshed = True

    def add_object(self,  colorpose: ColorPose) -> None:

        if len(self.objects) == 0:
            self.objects.append(ClusterObject(colorpose, 0))
            return

        distances = [
            a.get_distance_to_pose(colorpose.pose) for a in self.objects]

        index_min = np.argmin(distances)

        if distances[index_min] < self.threshold:
            self.objects[index_min].update_position(
                colorpose.pose, colorpose.color)
            return

        rospy.loginfo("New object added")

        self.objects.append(ClusterObject(colorpose, len(self.objects)))

    def draw_markers(self) -> None:
        stamp = rospy.Time(0)
        header = Header(stamp=stamp, frame_id='map')
        objects_len = len(self.objects)

        marker_arr = [Marker(header=header, action=Marker.DELETEALL)]
        id = 0

        for face in self.objects:

            if face.emmited == False:
                continue

            # print(face.picked_color)

            c = face.picked_color

            color_barva = rgba(c.r, c.g, c.b)

            p = face.pose.position
            text_pose = Pose(position=Point(x=p.x, y=p.y+0.1))

            marker_arr.append(Marker(header=header,
                                     pose=face.pose,
                                     type=Marker.SPHERE,
                                     action=Marker.ADD,
                                     frame_locked=False,
                                     id=id,
                                     scale=Vector3(0.1, 0.1, 0.1),
                                     color=color_barva))

            marker_arr.append(Marker(header=header,
                                     pose=text_pose,
                                     type=Marker.TEXT_VIEW_FACING,
                                     text=("Torus " + face.named_color),
                                     action=Marker.ADD,
                                     frame_locked=False,
                                     id=id + objects_len,
                                     scale=Vector3(0.1, 0.1, 0.1),
                                     color=color_barva))

            id += 1

        self.markers_pub.publish(marker_arr)

    def get_all_objects(self) -> "list[ClusterObject]":
        return self.objects


def main():
    rospy.init_node('cluster_database', anonymous=True)

    subscribe_topics = rospy.get_param("~topic_list")
    marker_topic = rospy.get_param("~marker_topic")
    number_of_objects = rospy.get_param("~number_of_objects")

    CluserDatabase(subscribe_topics=subscribe_topics,
                   marker_topic=marker_topic, threshold=0.7, number_of_objects=number_of_objects)

    rospy.spin()


if __name__ == '__main__':
    main()
