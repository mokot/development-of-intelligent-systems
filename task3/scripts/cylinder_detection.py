#!/usr/bin/python3
import rospy
import cv2
import tf2_ros
import numpy as np
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Pose, Vector3, Point, Quaternion, PointStamped
from std_msgs.msg import Header, ColorRGBA, String
import matplotlib.pyplot as plt
import tf2_geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray
from matplotlib.colors import rgb_to_hsv
from sound_play.libsoundplay import SoundClient


class Clustering:
    def __init__(self, threshold=0.5):
        self.clusters = []
        self.threshold = threshold

    def distance(self, pose1, pose2):
        return np.sqrt((pose1.x - pose2.x) ** 2 + (pose1.y - pose2.y) ** 2)

    def add(self, pose):
        new_cluster = True

        for cluster in self.clusters:
            for point in cluster:
                if self.distance(pose, point) < self.threshold:
                    cluster.append(pose)
                    new_cluster = False

                    if len(cluster) == 4:
                        return True

                    break
            if not new_cluster:
                break

        if new_cluster:
            self.clusters.append([pose])

        return False


class CylinderDetection:
    def __init__(self):
        rospy.init_node("cylinder_detection")
        self.bridge = CvBridge()

        self.scan_sub = message_filters.Subscriber("/arm_scan", LaserScan)
        self.image_sub = message_filters.Subscriber("/arm_camera/rgb/image_raw", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.scan_sub, self.image_sub], 10, 0.5
        )
        self.ts.registerCallback(self.process_scan)

        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        self.marker_pub = rospy.Publisher(
            "/cylinder_markers", MarkerArray, queue_size=10
        )
        self.finished_pub = rospy.Publisher(
            "task3/all_cylinders", String, queue_size=10
        )
        self.cylinder_pub = rospy.Publisher(
            "task3/cylinder_pose", Marker, queue_size=10
        )

        rospy.sleep(0.5)
        self.marker_pub.publish(
            [
                Marker(
                    header=Header(frame_id="map", stamp=rospy.Time(0)),
                    action=Marker.DELETEALL,
                )
            ]
        )
        self.markers = []

        self.found_cylinders = 0

        self.cluster = Clustering()

        self.soundhandle = SoundClient()
        self.voice = "voice_kal_diphone"
        self.volume = 1.0

        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            self.publish_markers()

            if self.found_cylinders > 3:
                self.finished_pub.publish("true")

            r.sleep()

        rospy.spin()

    def process_scan(self, scan, arm_image):
        self.N = len(scan.ranges)
        self.resolution = scan.range_max / self.N / 4

        image = np.zeros((self.N, self.N), np.uint8)

        angle = scan.angle_min
        for r in scan.ranges:
            angle += scan.angle_increment

            if np.isnan(r):
                continue

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            x_p, y_p = self.to_pixel_space(x, y)

            if x_p < 0 or x_p >= self.N or y_p < 0 or y_p >= self.N:
                continue

            image[y_p, x_p] = 255

        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            2,
            100,
            param1=10,
            param2=25,
            minRadius=30,
            maxRadius=35,
        )

        # debug_image = image.copy()
        # debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
        # if circles is not None:
        #     circles = np.uint16(np.around(circles))
        #     for i in circles[0, :]:
        #         cv2.circle(debug_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #         cv2.circle(debug_image, (i[0], i[1]), 2, (0, 0, 255), 3)

        # cv2.imshow("image", debug_image)
        # cv2.waitKey(1)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                x, y = self.to_world_space(i[0], i[1])

                point = PointStamped(
                    header=Header(
                        stamp=scan.header.stamp, frame_id="arm_camera_rgb_optical_frame"
                    ),
                    point=Point(x=-y, y=0, z=x),
                )

                try:
                    world_point = self.tf_buf.transform(point, "map")
                except:
                    world_point = None

                # To bo treba spremenit
                if world_point:
                    theta = np.arctan2(y, x)

                    image = self.bridge.imgmsg_to_cv2(arm_image, "rgb8")

                    cylinder_x = int(
                        np.interp(
                            -theta,
                            [scan.angle_min, scan.angle_max],
                            [0, image.shape[1]],
                        )
                    )

                    width = 50
                    x1, x2 = cylinder_x - width / 2, cylinder_x + width / 2

                    height = 100
                    offset = 150
                    y1, y2 = image.shape[0] - offset, image.shape[0] - offset + height

                    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

                    color = image[y1:y2, x1:x2]

                    hue = rgb_to_hsv(np.mean(color, axis=1))[0][0] * 360

                    color_name = ""
                    marker_color = ColorRGBA(r=0, g=0, b=0, a=1)
                    if hue < 15 or hue > 330:
                        color_name = "red"
                        marker_color = ColorRGBA(r=1, g=0, b=0, a=1)
                        # self.add_marker(world_point.point.x, world_point.point.y, ColorRGBA(r=1, g=0, b=0, a=1))
                    elif 40 < hue < 70:
                        color_name = "yellow"
                        marker_color = ColorRGBA(r=1, g=1, b=0, a=1)
                        # self.add_marker(world_point.point.x, world_point.point.y, ColorRGBA(r=1, g=1, b=0, a=1))
                    elif 90 < hue < 150:
                        color_name = "green"
                        marker_color = ColorRGBA(r=0, g=1, b=0, a=1)
                        # self.add_marker(world_point.point.x, world_point.point.y, ColorRGBA(r=0, g=1, b=0, a=1))
                    elif 190 < hue < 240:
                        color_name = "blue"
                        marker_color = ColorRGBA(r=0, g=0, b=1, a=1)
                        # self.add_marker(world_point.point.x, world_point.point.y, ColorRGBA(r=0, g=0, b=1, a=1))

                    if self.cluster.add(world_point.point):
                        self.add_marker(
                            world_point.point.x,
                            world_point.point.y,
                            marker_color,
                            color_name,
                        )
                        self.soundhandle.say(
                            "I see a " + color_name + " cylinder",
                            self.voice,
                            self.volume,
                        )

                        self.found_cylinders += 1

                    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    # cv2.waitKey(1)

    def to_pixel_space(self, x, y):
        x_p = int(x / self.resolution)
        y_p = int(y / self.resolution) + self.N // 2

        return x_p, y_p

    def to_world_space(self, x_p, y_p):
        x = x_p * self.resolution
        y = (y_p - self.N / 2) * self.resolution

        return x, y

    def publish_markers(self):
        self.marker_pub.publish(self.markers)

    def add_marker(self, x, y, color, color_name):
        pose = Pose(position=Point(x, y, 0), orientation=Quaternion(0, 0, 0, 1))

        marker = Marker(
            header=Header(frame_id="map", stamp=rospy.Time.now()),
            pose=pose,
            type=Marker.CYLINDER,
            action=Marker.ADD,
            id=len(self.markers),
            lifetime=rospy.Time(0),
            color=color,
            text=color_name,
            scale=Vector3(0.2, 0.2, 0.2),
        )

        print("CYLINDER:::", marker)
        self.cylinder_pub.publish(marker)

        self.markers.append(marker)


if __name__ == "__main__":
    CylinderDetection()
