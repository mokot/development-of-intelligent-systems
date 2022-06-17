#!/usr/bin/python3

import math
import rospy
import dlib
import cv2
import numpy as np
import tf2_ros
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose, Point
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
import face_recognition
import numpy as np
import json

from pathlib import Path


# Keep this
import tf2_geometry_msgs
import sys
from turtle import pos, position, update

#  Our message type
from task1.msg import FaceLocation


def rgba(r, g, b):
    return ColorRGBA(r / 255, g / 255, b / 255, 1)


color_array = [
    rgba(245, 40, 145),
    rgba(140, 39, 245),
    rgba(245, 196, 39),
    rgba(80, 245, 39),
    rgba(26, 203, 194),
    rgba(26, 72, 203),
    rgba(255, 23, 23)
]


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class PotentialHits:
    def __init__(self, threshold=0.5):
        # Locations of type (float, float) as in x, y point
        self.locations = []
        self.threshold = threshold

    def add(self, n_l):
        """_summary_

        Args:
            n_l ((float, float)): Point to add and compare to current base

        Returns:
            boolean: _description_
        """
        to_add = True

        for l in self.locations:
            if math.sqrt((l[0] - n_l[0])**2 + (l[1] - n_l[1])**2) < self.threshold:
                to_add = False
                break

        if to_add == True:
            self.locations.append(n_l)

        return to_add


class Face:
    def __init__(self, encoding, pose, id, name):
        self.encoding = encoding
        self.pose = pose
        self.update_counter = 1
        self.id = id
        self.name = name

    def update_position(self, update_pose):
        """Updates the position of a face using a weight. The weight is tracked internally.

        Args:
            update_pose (Pose): _description_
        """
        c_p = self.pose.position
        u_p = update_pose.position
        w = self.update_counter

        self.pose = Pose(position=Point(x=(c_p.x * w + u_p.x) / (w+1),
                                        y=(c_p.y * w + u_p.y) / (w+1),
                                        z=(c_p.z * w + u_p.z) / (w+1)))

        print(f'{Colors.OKBLUE}> Updated position of a face {Colors.ENDC}')

        self.update_counter += 1

    def get_distance_to_pose(self, pose):
        c_p = self.pose.position
        o_p = pose.position

        return math.sqrt((c_p.x - o_p.x)**2 + (c_p.y - o_p.y)**2)


class FaceDatabase:
    def __init__(self):
        self.faces = []

        # Just for fun :)
        self.lookup_names = []
        self.lookup_encodings = []

        # We generated a lookup table for already known faces, so we can attach some names to them
        path = Path(__file__).parent / "../data/encoding_data.json"

        with open(path) as json_file:
            data = json.load(json_file)

            for key in data:
                self.lookup_names.append(key["name"])
                self.lookup_encodings.append(key["encoding"])

    def add_face(self, encoding, pose):
        try:
            faces_to_compare = [
                a for a in self.faces if a.get_distance_to_pose(pose) < 1]

            comparison_arr = face_recognition.face_distance(
                [x.encoding for x in faces_to_compare], encoding)

            index_min = np.argmin(comparison_arr)

            face = faces_to_compare[index_min]

            face.update_position(pose)

            return None

        except ValueError:
            print(
                f'{Colors.OKGREEN}{Colors.BOLD}> No face with this encoding has ever been matched, adding to database {Colors.ENDC}')

            # Euclidan distances of comparisons -> Smallest, the closest to our face
            lookup_comparison = face_recognition.face_distance(
                self.lookup_encodings, encoding)

            index_min = np.argmin(lookup_comparison)

            name = self.lookup_names[index_min]

            id = len(self.faces)

            face = Face(encoding, pose, id, name)

            self.faces.append(face)

            return face

    def get_all_faces(self):
        return self.faces


class FaceLocalizer:
    def __init__(self):

        rospy.init_node('face_localizer', anonymous=True)

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        self.face_detector = dlib.get_frontal_face_detector()

        # A help variable for holding the dimensions of the image
        self.dims = (0, 0, 0)

        self.show_faces = False

        self.face_database = FaceDatabase()

        self.hits = PotentialHits()

        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher(
            'face_markers', MarkerArray, queue_size=1000)

        # Publisher for probable face locations
        self.potential_faces_pub = rospy.Publisher(
            '/task1/potential_face', FaceLocation, queue_size=1000)

        # Publisher for definite face locations
        self.faces_pub = rospy.Publisher(
            '/task1/face', FaceLocation, queue_size=1000)

        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

    def get_pose(self, coords, dist, stamp):
        # Calculate the position of the detected face

        k_f = 554  # kinect focal length in pixels

        x1, x2, y1, y2 = coords

        face_x = self.dims[1] / 2 - (x1+x2)/2.

        angle_to_target = np.arctan2(face_x, k_f)

        # Get the angles in the base_link relative coordinate system
        x, y = dist*np.cos(angle_to_target), dist*np.sin(angle_to_target)

        # Define a stamped message for transformation - in the "camera rgb frame"
        point_s = PointStamped(header=Header(stamp=stamp, frame_id='camera_rgb_optical_frame'),
                               point=Point(x=-y, y=0, z=x))

        print(point_s)

        # Get the point in the "map" coordinate system
        try:
            point_world = self.tf_buf.transform(point_s, "map")
            # Create a Pose object with the same position
            pose = Pose(position=point_world.point)

        except Exception:
            pose = None

        return pose

    def add_markers(self):
        """Removed all markers and reads one for all faces
        """

        global color_array

        faces = self.face_database.get_all_faces()

        stamp = rospy.Time(0)
        header = Header(stamp=stamp, frame_id='map')

        marker_arr = [Marker(header=header, action=Marker.DELETEALL)]
        id = 0

        for face in faces:

            p = face.pose.position
            text_pose = Pose(position=Point(x=p.x, y=p.y+0.1))
            color = color_array[id % len(color_array)]

            marker_arr.append(Marker(header=header,
                                     pose=face.pose,
                                     type=Marker.SPHERE,
                                     action=Marker.ADD,
                                     frame_locked=False,
                                     id=id,
                                     scale=Vector3(0.1, 0.1, 0.1),
                                     color=color))

            marker_arr.append(Marker(header=header,
                                     pose=text_pose,
                                     type=Marker.TEXT_VIEW_FACING,
                                     text=(face.name),
                                     action=Marker.ADD,
                                     frame_locked=False,
                                     id=id + len(faces),
                                     scale=Vector3(0.1, 0.1, 0.1),
                                     color=color))

            id += 1

        self.markers_pub.publish(marker_arr)

    def find_faces(self):
        # Get the next rgb and depth images that are posted from the camera
        try:
            rgb_image_message = rospy.wait_for_message(
                "/camera/rgb/image_raw", Image)
        except Exception as e:
            print(e)
            return 0

        try:
            depth_image_message = rospy.wait_for_message(
                "/camera/depth/image_raw", Image)
        except Exception as e:
            print(e)
            return 0

        # Convert the images into a OpenCV (numpy) format
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_message, "bgr8")
        except CvBridgeError as e:
            print(e)

        try:
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_image_message, "32FC1")
        except CvBridgeError as e:
            print(e)

        # Set the dimensions of the image
        self.dims = rgb_image.shape

        # Detect the faces in the image
        face_rectangles = self.face_detector(rgb_image, 0)

        # For each detected face, extract the depth from the depth image
        for face_rectangle in face_rectangles:
            print('> Faces were detected')

            padding = 20

            # The coordinates for
            x1 = face_rectangle.left()
            x2 = face_rectangle.right()
            y1 = face_rectangle.top()
            y2 = face_rectangle.bottom()

            # The coordinates of the picture
            p_x1 = max(x1 - padding, 0)
            p_x2 = min(x2 + padding, self.dims[1] - 1)
            p_y1 = max(y1 - 50, 0)
            p_y2 = min(y2 + padding, self.dims[0] - 1)

            # Extract region containing face
            face_region = rgb_image[p_y1:p_y2, p_x1:p_x2]

            if self.show_faces:
                try:
                    resized_face = cv2.resize(
                        rgb_image[y1:y2, x1:x2], (face_rectangle.width() * 5, face_rectangle.height() * 5))

                    cv2.imshow("output", resized_face)
                    cv2.waitKey(1)
                except Exception as e:
                    print("> Couldn't show image")

            # Find the distance to the detected face
            face_distance = float(np.nanmean(depth_image[y1:y2, x1:x2]))

            # print(f'Distance to face is {face_distance}')

            # Get the time that the depth image was recieved
            depth_time = depth_image_message.header.stamp

            # Find the location of the detected face
            pose = self.get_pose((x1, x2, y1, y2), face_distance, depth_time)

            if pose is None:
                continue

            pos = pose.position

            if math.isnan(pos.x) or math.isnan(pos.y):
                continue

            hit = self.hits.add((pos.x, pos.y))

            if hit:
                print(f'> Published POTENTIAL position is {[pos.x, pos.y]}')
                # Publish potential face
                self.potential_faces_pub.publish(FaceLocation(
                    id=0, x=pose.position.x, y=pose.position.y, name=""))

            # Returns the number of face encoding the model finds.
            # If length = 0, then we found 0 faces, else we found atlest 1
            # In our case that just means we definetly found a face.
            boxes = face_recognition.face_locations(face_region)

            encoding = face_recognition.face_encodings(
                face_region, boxes, num_jitters=3, model="large")

            if len(encoding) == 0:
                print(
                    f'{Colors.WARNING}> dLib found a face, but face_recognizer didn\'t recognize it')
                continue
            else:
                encoding = encoding[0]

            face = self.face_database.add_face(encoding, pose)

            if face is not None:
                self.faces_pub.publish(FaceLocation(
                    id=face.id, x=pos.x, y=pos.y, name=face.name))

                print(
                    f'{Colors.BOLD}{Colors.OKCYAN}> Published position is {[pos.x, pos.y]} with id {face.id} (name {face.name})  {Colors.ENDC}')

            self.add_markers()


def main():
    face_finder = FaceLocalizer()

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        face_finder.find_faces()
        rate.sleep()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
