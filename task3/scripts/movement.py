#!/usr/bin/python3
import rospy
import actionlib
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from sklearn.cluster import KMeans
from dataclasses import dataclass
from sensor_msgs.msg import Image
from tf.transformations import quaternion_from_euler
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, Vector3, PoseStamped, Point, Quaternion, Twist
from visualization_msgs.msg import Marker, MarkerArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import String, Header, ColorRGBA
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetPlan
from sound_play.libsoundplay import SoundClient
from task1.srv import Normal_vector
from task3.srv import speech


PARKING_NUM = 250


class State(Enum):
    IDLE = 0
    EXPLORING = 1
    PARKING = 2
    GOTO_PARKING = 3
    SCANNING = 4
    SERVING = 5
    CYLINDER = 6
    SERVING_FOOD = 7


@dataclass
class Goal_Pose:
    x: float
    y: float

    rot_z: float
    rot_w: float

    def to_base_goal(self):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.orientation.z = self.rot_z
        goal.target_pose.pose.orientation.w = self.rot_w
        goal.target_pose.pose.position.x = self.x
        goal.target_pose.pose.position.y = self.y

        return goal

    def to_pose_stamped(self):
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.orientation.z = self.rot_z
        goal.pose.orientation.w = self.rot_w
        goal.pose.position.x = self.x
        goal.pose.position.y = self.y
        goal.header.stamp = rospy.Time.now()

        return goal


status_to_text = {
    0: "PENDING",
    1: "ACTIVE",
    2: "PREEMPTED",
    3: "SUCCEEDED",
    4: "ABORTED",
    5: "REJECTED",
    6: "PREEMPTING",
    7: "RECALLING",
    8: "RECALLED",
    9: "LOST",
}


class MovementNode:
    def __init__(self):
        rospy.init_node("movebase_client_py")

        self.markers = []
        self.marker_pub = rospy.Publisher("/goal_markers", MarkerArray, queue_size=10)
        self.publish_twist = rospy.Publisher(
            "/cmd_vel_mux/input/teleop", Twist, queue_size=1000
        )
        self.arm_controller_pub = rospy.Publisher(
            "/task3/arm_controller/command", String, queue_size=10
        )

        self.marker_pub.publish(
            [
                Marker(
                    header=Header(frame_id="map", stamp=rospy.Time(0)),
                    action=Marker.DELETEALL,
                )
            ]
        )
        self.plan_checker = rospy.ServiceProxy("/move_base/make_plan", GetPlan)

        self.positions = self.generate_goals(8)
        self.arm_controller_pub.publish("setup_scanner")

        self.soundhandle = SoundClient()
        rospy.sleep(1)
        self.voice = "voice_kal_diphone"
        self.volume = 1.0
        # self.soundhandle.say("Hello world, nice to meet you!", self.voice, self.volume)

        self.goal_count = 0

        self.goal_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.goal_client.wait_for_server()

        self.cancel_sub = rospy.Subscriber(
            "task3/command", String, self.command_callback
        )
        self.bridge = CvBridge()

        self.circle_sub = rospy.Subscriber(
            "/task3/all_circles", Pose, self.circle_callback
        )
        self.cylinder_sub = rospy.Subscriber(
            "/task3/all_cylinders", String, self.cylinder_callback
        )

        # self.cylinder_one_sub = rospy.Subscriber(
        #     "/task3/cylinder_pose", Marker, self.cylinder_one_callback
        # )

        self.found_cylinders = False
        self.parking_pose = None
        self.parking_num = 0
        self.serving_pose = None
        self.serve_num = 0
        self.food_or_name = ""
        self.cylinder_pose = None
        self.goal_food_client = []

        rospy.wait_for_service("normal_vector")
        self.vector_service = rospy.ServiceProxy("normal_vector", Normal_vector)

        self.final_names, self.final_food = [], []
        self.qr_sub = rospy.Subscriber("/qr/data", String, self.qr_callback)

        # DEBUGGING
        # ["name", "x", "y", "z", "w"]
        self.faces = np.asarray(
            [
                ["Ana", 0.098, -0.185, -0.006, 0.999],
                ["Nina", -0.092, -1.533, -0.700, 0.714],
                ["Irena", 3.677, -0.798, 0.027, 0.999],
                ["Mateja", 0.993, 1.143, 0.999, 0.004],
                ["Gargamel", -0.161, 2.721, 0.701, 0.713],
            ]
        )
        # self.faces = np.asarray(
        #     [
        #         ["Gargamel", 1.237, 2.720, 0.711, 0.703],
        #         ["Mateja", -1.555, 1.595, 0.999, 0.012],
        #         ["Nina", -0.275, -1.514, -0.726, 0.688],
        #         ["Irena", 3.694, -0.746, 0.136, 0.991],
        #         ["Ana", 0.017, -0.147, -0.019, 0.999],
        #     ]
        # )
        self.face_sub = rospy.Subscriber(
            "/face_markers", MarkerArray, self.faces_callback
        )

        # DEBUGGING
        # ["color", "food", "x", "y", "z", "w"]
        self.food = np.asarray(
            [
                ["red", "torta", -1.258, -0.121, -0.713, 0.701],
                ["yellow", "baklava", 3.408, -1.123, -0.648, 0.762],
                ["blue", "pomfri", 1.010, -0.159, -0.913, 0.408],
                ["green", "pizza", 1.222, 2.547, 0.704, 0.710],
            ]
        )
        # self.food = np.asarray(
        #     [
        #         ["red", "torta", -1.248, -0.146, -0.705, 0.709],
        #         ["yellow", "solata", 2.048, -1.478, -0.697, 0.717],
        #         ["blue", "pomfri", 1.142, -0.175, -0.706, 0.708],
        #         ["green", "pizza", 2.051, 2.554, 0.705, 0.709],
        #     ]
        # )
        self.temp_detected_food = ""

        # ASR
        rospy.wait_for_service("automated_speech_recognition")
        self.asr = rospy.ServiceProxy("automated_speech_recognition", speech)
        
        # self.asr(
        #     "Here is your %s." % "pizza"
        # )
        # payment_method = self.asr(
        #     "Will you pay by cash or credit card?"
        # ).data
        # payment_method = "CASH" if "cas" in payment_method else "CARD"
        # print("::: payment method: ", payment_method)
        # if payment_method == "CASH":
        #     self.arm_controller_pub.publish("move_left")
        #     rospy.sleep(1)
        #     self.arm_controller_pub.publish("still_scanner")
        # else:
        #     self.arm_controller_pub.publish("move_right")
        #     rospy.sleep(1)
        #     self.arm_controller_pub.publish("still_scanner")

        # rating = self.asr(
        #     "How satisfied were you with the service on the scale from 1 to 5?"
        # ).data
        # try:
        #     rating = self.rating_scale[rating]
        # except:
        #     rating = -1
        # print("::: rating: ", rating)
        # self.soundhandle.say(
        #     "Thank you and goodbye.",
        #     self.voice,
        #     self.volume,
        # )

        self.state = State.EXPLORING
        # self.arm_controller_pub.publish("park")

        self.publish_goal()

        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.publish_markers()

            if self.state == State.PARKING:
                self.park()

            rate.sleep()

    def circle_callback(self, msg):
        if self.state != State.EXPLORING:
            return

        rospy.loginfo("Found all circles")
        self.parking_pose = msg.position
        self.circle_sub.unregister()

        if self.found_cylinders:
            self.cylinder_sub.unregister()
            self.initiate_park()

    def cylinder_callback(self, msg):
        if self.state != State.EXPLORING:
            return

        rospy.loginfo("Found all cylinders")
        self.found_cylinders = True
        self.cylinder_sub.unregister()

        if self.parking_pose:
            self.circle_sub.unregister()
            self.initiate_park()

    def cylinder_one_callback(self, msg):
        if self.state != State.EXPLORING:
            return

        marker_color = msg.text
        food_index = np.where(self.food[:, 0] == marker_color)[0]
        if len(food_index):
            food_index = food_index[0]
            
            print(":::", self.food[food_index])
            self.temp_detected_food = self.food[food_index, 1]
            
            self.cylinder_pose = Goal_Pose(
                x=float(self.food[food_index, 2]),
                y=float(self.food[food_index, 3]),
                rot_z=float(self.food[food_index, 4]),
                rot_w=float(self.food[food_index, 5]),
            )
            self.find_cylinder()

    def find_cylinder(self):
        self.goal_client.cancel_all_goals()
        self.goal_client.cancel_goal()
        self.state = State.CYLINDER
        self.arm_controller_pub.publish("food_detection")

        self.goal_client.send_goal(
            self.cylinder_pose.to_base_goal(), done_cb=self.done_callback
        )

    def qr_callback(self, msg):
        if self.state == State.SCANNING:
            self.soundhandle.say("Scanning", self.voice, self.volume)
            rospy.loginfo("Found QR code: %s" % msg.data)

            self.goal_client.cancel_all_goals()
            self.arm_controller_pub.publish("still_scanner")

            codes = msg.data.split(", ")

            self.final_names, self.final_food = [], []
            for code in codes:
                temp_name, temp_food = code.split(" ")
                self.final_names.append(temp_name)
                self.final_food.append(temp_food)

            print(self.final_names, self.final_food)

            self.state = State.SERVING
            self.serve_num = 0
            self.goal_food_client = []

            for temp_food, temp_name in zip(self.final_food, self.final_names):
                food_index = np.where(self.food[:, 1] == temp_food)[0]
                face_index = np.where(self.faces[:, 0] == temp_name)[0]

                if len(food_index) and len(face_index):
                    food_index = food_index[0]
                    # rospy.loginfo("Serving %s" % temp_food)
                    self.food_or_name = temp_food
                    serving_pose = Goal_Pose(
                        x=float(self.food[food_index, 2]),
                        y=float(self.food[food_index, 3]),
                        rot_z=float(self.food[food_index, 4]),
                        rot_w=float(self.food[food_index, 5]),
                    )
                    self.goal_food_client.append(serving_pose)

                    face_index = face_index[0]
                    # rospy.loginfo("Serving %s" % temp_name)
                    self.food_or_name = temp_name
                    serving_pose = Goal_Pose(
                        x=float(self.faces[face_index, 1]),
                        y=float(self.faces[face_index, 2]),
                        rot_z=float(self.faces[face_index, 3]),
                        rot_w=float(self.faces[face_index, 4]),
                    )
                    self.goal_food_client.append(serving_pose)

            print(self.goal_food_client)
            self.publish_food_client_goal()

    def serveFood_callback(self):
        if len(self.final_food) <= self.serve_num or self.state != State.SERVING_FOOD:
            return

        # Find food else continue
        food = self.final_food[self.serve_num]
        food_index = np.where(self.food[:, 1] == food)[0]
        if len(food_index):
            food_index = food_index[0]
            rospy.loginfo("Serving %s" % food)
            self.food_or_name = food
            self.serving_pose = Goal_Pose(
                x=float(self.food[food_index, 2]),
                y=float(self.food[food_index, 3]),
                rot_z=float(self.food[food_index, 4]),
                rot_w=float(self.food[food_index, 5]),
            )
            self.find_food()
        else:
            self.serve_num += 1
            self.serveFood_callback()

    def serveFace_callback(self):
        if len(self.final_names) <= self.serve_num or self.state != State.SERVING:
            return

        # Find face else continue
        name = self.final_names[self.serve_num]
        face_index = np.where(self.faces[:, 0] == name)[0]
        if len(face_index):
            face_index = face_index[0]
            rospy.loginfo("Serving %s" % name)
            self.food_or_name = name
            self.serving_pose = Goal_Pose(
                x=float(self.faces[face_index, 1]),
                y=float(self.faces[face_index, 2]),
                rot_z=float(self.faces[face_index, 3]),
                rot_w=float(self.faces[face_index, 4]),
            )
            self.find_face()

        self.serve_num += 1

    def faces_callback(self, msg):
        if self.state != State.EXPLORING and self.state != State.PARKING:
            return

        markers = msg.markers
        faces = []
        for marker in markers:
            if marker.text:
                faces.append(
                    [marker.text, marker.pose.position.x, marker.pose.position.y]
                )
        print("Faces:", faces)
        faces = np.asarray(faces)

    def find_food(self):
        self.goal_client.cancel_all_goals()

        self.goal_client.send_goal(
            self.serving_pose.to_base_goal(), done_cb=self.done_callback
        )

    def find_face(self):
        self.goal_client.cancel_all_goals()

        self.goal_client.send_goal(
            self.serving_pose.to_base_goal(), done_cb=self.done_callback
        )

    def initiate_park(self):
        self.goal_client.cancel_all_goals()
        self.goal_client.cancel_goal()
        self.state = State.GOTO_PARKING
        self.parking_num = 0

        angle = self.vector_service(self.parking_pose.x, self.parking_pose.y).angle

        x_diff = np.cos(angle)
        y_diff = np.sin(-angle)
        yaw = angle + np.pi

        target_x = self.parking_pose.x + x_diff * 0.5
        target_y = self.parking_pose.y + y_diff * 0.5

        quat = quaternion_from_euler(0, 0, yaw)

        target = Goal_Pose(x=target_x, y=target_y, rot_z=quat[2], rot_w=quat[3])

        self.goal_client.send_goal(target.to_base_goal(), done_cb=self.done_callback)

    def publish_markers(self):
        self.marker_pub.publish(self.markers)

    def add_marker(self, x, y):
        pose = Pose(position=Point(x, y, 0), orientation=Quaternion(0, 0, 0, 1))

        marker = Marker(
            header=Header(frame_id="map", stamp=rospy.Time.now()),
            pose=pose,
            type=Marker.CUBE,
            action=Marker.ADD,
            id=len(self.markers),
            lifetime=rospy.Time(0),
            color=ColorRGBA(0, 0, 1, 1),
            scale=Vector3(0.2, 0.2, 0.2),
        )

        self.markers.append(marker)

    def add_arrow_marker(self, x, y, rot_z, rot_w):
        pose = Pose(position=Point(x, y, 0), orientation=Quaternion(0, 0, rot_z, rot_w))

        marker = Marker(
            header=Header(frame_id="map", stamp=rospy.Time.now()),
            pose=pose,
            type=Marker.ARROW,
            action=Marker.ADD,
            id=len(self.markers),
            lifetime=rospy.Time(0),
            color=ColorRGBA(0, 1, 0, 1),
            scale=Vector3(0.5, 0.05, 0.05),
        )

        self.markers.append(marker)

    def park(self):
        try:
            image_message = rospy.wait_for_message("/arm_camera/rgb/image_raw", Image)
        except Exception as e:
            print(e)
            return

        try:
            image = self.bridge.imgmsg_to_cv2(image_message, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[:360, :]

        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1=30,
            param2=10,
            minRadius=0,
            maxRadius=25,
        )

        # if circles is not None:
        #     circles = np.uint16(np.around(circles))
        #     for i in circles[0, :]:
        #         # draw the outer circle
        #         cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #         # draw the center of the circle
        #         cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

        if circles is None:
            return

        # TODO: Če bo problem večih cirklov

        circle = circles[0][0]
        center = image.shape[1] // 2

        twist = Twist()
        twist.linear.x = 0.15
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0

        center_margin = 30
        forward_margin = 30
        if circle[0] - center > center_margin and self.parking_num < PARKING_NUM:
            print(self.parking_num, "- Turning right", circle)
            twist.angular.z = -1
            twist.linear.x = 0

        elif circle[0] - center < -center_margin and self.parking_num < PARKING_NUM:
            print(self.parking_num, "- Turning left", circle)
            twist.angular.z = 1
            twist.linear.x = 0

        elif (
            circle[1] < image.shape[0] - forward_margin
            and self.parking_num < PARKING_NUM
        ):
            print(self.parking_num, "- Moving forward", circle)
            twist.linear.x = 0.15
        else:
            # self.soundhandle.say("Time to self destruct", self.voice, self.volume)
            print(self.parking_num, "- Moving backward", circle)
            self.state = State.SCANNING
            twist.linear.x = -0.50
            # self.arm_controller_pub.publish("self_destruct")

        self.parking_num += 1
        self.publish_twist.publish(twist)

    def check_plan(self, start, goal):
        plan = self.plan_checker(start, goal, 0.5)
        return plan

    def generate_goals(self, goal_count):
        map_data = rospy.wait_for_message("map", OccupancyGrid)

        scale_factor = 4

        position = map_data.info.origin.position
        resolution = map_data.info.resolution

        # Read the map
        directory = os.path.dirname(__file__)
        whole_map = cv2.imread(
            os.path.join(directory, "../map/edited_map.pgm"), cv2.IMREAD_GRAYSCALE
        )

        # Scale it down so that KMeans takes less time
        scaled_map = cv2.resize(
            whole_map, (0, 0), fx=1 / scale_factor, fy=1 / scale_factor
        )

        # Make a binary map
        roaming_area = np.where(scaled_map == 254, 1, 0).astype(np.uint8)

        # Erode the map a little
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roaming_area = cv2.erode(roaming_area, kernel)

        # plt.imshow(roaming_area)
        # plt.show()

        # Calculate the centroids
        roaming_area = list(zip(*np.nonzero(roaming_area)))
        kmeans = KMeans(n_clusters=goal_count).fit(roaming_area)

        # Rescale goals back to original size
        goals = scale_factor * kmeans.cluster_centers_
        goals[:, [1, 0]] = goals[:, [0, 1]]

        # plt.imshow(whole_map, cmap='gray')
        # plt.scatter(goals[:, 0], goals[:, 1], c='r', s=100)
        # plt.show()

        # Move into the navigation frame
        goals[:, 0] = goals[:, 0] * resolution + position.x
        goals[:, 1] = (whole_map.shape[1] - goals[:, 1]) * resolution + position.y

        goals = [Goal_Pose(x, y, 0, 1) for x, y in goals]

        def get_closest_goal(goal):
            closest = None
            min_dist = 99999

            for g in goals:
                dist = len(
                    self.check_plan(
                        g.to_pose_stamped(), goal.to_pose_stamped()
                    ).plan.poses
                )
                if dist < min_dist:
                    closest = g
                    min_dist = dist

            return closest

        # Get the shortest path through all goals
        start = get_closest_goal(Goal_Pose(0, 0, 0, 1))
        ordered_goals = [start]
        goals.remove(start)

        while len(goals) > 0:
            prev_goal = ordered_goals[-1]
            closest_next = get_closest_goal(prev_goal)
            goals.remove(closest_next)

            ordered_goals.append(closest_next)

        # Create goal poses including rotations
        final_goals = self.recalculate_angles(ordered_goals)

        return final_goals

    def recalculate_angles(self, ordered_goals):
        final_goals = []
        self.markers = []
        self.marker_pub.publish(
            [
                Marker(
                    header=Header(frame_id="map", stamp=rospy.Time(0)),
                    action=Marker.DELETEALL,
                )
            ]
        )

        for i in range(len(ordered_goals)):
            goal = ordered_goals[i]
            next_goal = ordered_goals[(i + 1) % len(ordered_goals)]

            x, y = goal.x, goal.y

            dx = next_goal.x - x
            dy = next_goal.y - y

            yaw = np.arctan2(dy, dx)
            quat = quaternion_from_euler(0, 0, yaw)
            rot_z, rot_w = quat[2], quat[3]

            final_goals.append(Goal_Pose(x, y, rot_z, rot_w))
            self.add_arrow_marker(x, y, rot_z, rot_w)

        return final_goals

    def publish_goal(self):
        if self.state != State.EXPLORING:
            return

        goal = self.positions[self.goal_count]
        rospy.loginfo(f"Sending new goal to {goal.x} {goal.y}!")
        self.goal_client.send_goal(goal.to_base_goal(), done_cb=self.done_callback)

    def publish_food_client_goal(self):
        if self.state != State.SERVING or self.serve_num >= len(self.goal_food_client):
            return

        goal = self.goal_food_client[self.serve_num]
        rospy.loginfo(f"Sending new goal to {goal.x} {goal.y}!")
        self.goal_client.send_goal(goal.to_base_goal(), done_cb=self.done_callback)

    def publish_specific_goal(self, goal):
        if self.state != State.SERVING:
            return

        rospy.loginfo(f"Sending new goal to {goal.x} {goal.y}!")
        self.goal_client.send_goal(goal.to_base_goal(), done_cb=self.done_callback)

    def done_callback(self, status, result):
        rospy.loginfo(f"Callback called with status {status_to_text[status]}")

        if self.state == State.EXPLORING:
            if status == 3:
                self.goal_count += 1
                if self.goal_count >= len(self.positions):
                    self.goal_count = 0
                    self.soundhandle.say("Reversing direction", self.voice, self.volume)
                    self.positions = self.positions[::-1]
                    self.positions = self.recalculate_angles(self.positions)
                self.publish_goal()

        if self.state == State.GOTO_PARKING:
            if status == 3:
                self.state = State.PARKING
                self.soundhandle.say("Parking", self.voice, self.volume)
                self.arm_controller_pub.publish("park")
                rospy.sleep(2)
                self.park()

        # if self.state == State.SERVING:
        #     if status == 3:
        #         self.soundhandle.say(
        #             "Hello %s! Here is your food." % self.food_or_name,
        #             self.voice,
        #             self.volume,
        #         )
        #         rospy.sleep(2)
        #         self.state = State.SERVING_FOOD
        #         self.serveFood_callback()

        # if self.state == State.SERVING_FOOD:
        #     if status == 3:
        #         self.soundhandle.say(
        #             "Serve %s" % self.food_or_name,
        #             self.voice,
        #             self.volume,
        #         )
        #         rospy.sleep(2)
        #         self.state = State.SERVING
        #         self.serveFace_callback()

        if self.state == State.CYLINDER:
            if status == 3:
                # TODO: recognize food
                print("Recognize food: %s" % self.temp_detected_food)
                rospy.sleep(1)
            self.state = State.EXPLORING
            self.arm_controller_pub.publish("setup_scanner")
            self.publish_goal()

        if self.state == State.SERVING:
            if status == 3:
                if self.serve_num < len(self.goal_food_client):
                    if self.serve_num % 2 == 0:
                        # Pick up the food
                        print("Pick up the food")
                        print("Serving to %s" % self.final_names[self.serve_num // 2])
                        self.soundhandle.say(
                            "Serving to %s" % self.final_names[self.serve_num // 2],
                            self.voice,
                            self.volume,
                        )
                    else:
                        # Serve to client
                        print("Serve to client")
                        print("Here is your %s" % self.final_food[self.serve_num // 2])
                        self.rating_scale = {
                            "one": 1,
                            "two": 2,
                            "three": 3,
                            "four": 4,
                            "five": 5,
                            "1": 1,
                            "2": 2,
                            "3": 3,
                            "4": 4,
                            "5": 5,
                        }

                        self.asr(
                            "Here is your %s." % self.final_food[self.serve_num // 2]
                        )
                        payment_method = self.asr(
                            "Will you pay by cash or credit card?"
                        ).data
                        payment_method = "CASH" if "cas" in payment_method else "CARD"
                        print("::: payment method: ", payment_method)
                        if payment_method == "CASH":
                            self.arm_controller_pub.publish("move_left")
                            self.arm_controller_pub.publish("still_scanner")
                        else:
                            self.arm_controller_pub.publish("move_right")
                            self.arm_controller_pub.publish("still_scanner")

                        rating = self.asr(
                            "How satisfied were you with the service on the scale from 1 to 5?"
                        ).data
                        try:
                            rating = self.rating_scale[rating]
                        except:
                            rating = -1
                        print("::: rating: ", rating)
                        self.soundhandle.say(
                            "Thank you and goodbye.",
                            self.voice,
                            self.volume,
                        )

                    self.serve_num += 1
                    self.publish_food_client_goal()

    def command_callback(self, msg):
        command = msg.data

        rospy.loginfo("Received command {command}")

        if command == "cancel":
            self.goal_client.cancel_goal()
            rospy.loginfo("Cancelled goal!")
        elif command == "next":
            self.publish_goal()
        elif command == "park":
            self.state = State.PARKING
        elif command == "scan":
            self.state = State.SCANNING


if __name__ == "__main__":
    MovementNode()
