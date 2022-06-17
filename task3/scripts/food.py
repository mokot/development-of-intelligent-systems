#!/usr/bin/python3
from cmath import nan
from pathlib import Path
import rospy
import cv2
import tf2_ros
import numpy as np
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Pose, Vector3, Point, Quaternion, PointStamped
from std_msgs.msg import Header, ColorRGBA, String, Int32
import matplotlib.pyplot as plt
import tf2_geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray
from matplotlib.colors import rgb_to_hsv
from sound_play.libsoundplay import SoundClient
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, PoseArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import PIL
import torch
from torchvision import datasets, models, transforms
from tf.transformations import quaternion_from_euler
from task1.srv import Normal_vector, Normal_vectorResponse


class FoodDetection:
    def __init__(self):
        rospy.init_node("food_detection")
        self.bridge = CvBridge()

        self.image_sub = message_filters.Subscriber("/arm_camera/rgb/image_raw", Image)
        self.arm_pub = rospy.Publisher(
            "/turtlebot_arm/arm_controller/command", JointTrajectory, queue_size=1
        )

    def set_arm(self, angles, duration=1):
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
        self.arm_pub.publish(trajectory)

    def recognize(self, image):
        input_size = 224

        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }
        class_dict = {0: "baklava", 1: "pizza", 2: "pomfri", 3: "solata", 4: "torta"}

        model_path = "./task3/food_rec/Foodero_dataset/Foodero/best_foodero_model.pt"

        model = torch.load(model_path)
        model.eval()

        img_p = PIL.Image.open(image)

        img = data_transforms["train"](img_p).unsqueeze(0)
        pred = model(img)

        pred_np = pred.cpu().detach().numpy().squeeze()
        class_ind = np.argmax(pred_np)
        if pred_np[class_ind] > 6:
            return class_dict[class_ind]

    def find_food(self):
        self.set_arm([0, 0.3, 1, -0.5], 1)

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

        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(img_hsv, (0, 0, 40), (0, 100, 150))
        # mask_down = ~cv2.inRange(cv_image_half_down_hsv, (0, 0, 150), (0, 100, 255))
        output = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
        output[np.where((output == [0, 0, 0]).all(axis=-1))] = [255, 255, 255]

        # Tranform image to grayscale
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        # Do histogram equlization
        img = cv2.equalizeHist(gray)
        # Binarize the image, there are different ways to do it
        # ret, thresh = cv2.threshold(img, 50, 255, 0)
        # ret, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 25
        )

        # Extract contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Example how to draw the contours, only for visualization purposes
        cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
        cv2.imshow("Contours", img)
        cv2.waitKey(1)
        # Fit elipses to all extracted contours
        elps = []
        for cnt in contours:
            if cnt.shape[0] >= 10:
                ellipse = cv2.fitEllipse(cnt)
                elps.append(ellipse)

        # print(elps)
        # Find two elipses with same centers
        candidates = []
        for n in range(len(elps)):
            for m in range(n + 1, len(elps)):
                e1 = elps[n]
                e2 = elps[m]
                # print(e1)
                # print(e2)
                dist = np.sqrt(
                    ((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2)
                )
                center = (int(e1[0][1]), int(e1[0][0]))
                size1 = (e1[1][0] + e1[1][1]) / 2
                el_var1 = abs(e1[1][0] - e1[1][1])
                size2 = (e2[1][0] + e2[1][1]) / 2
                el_var2 = abs(e2[1][0] - e2[1][1])
                size_diff = abs(size1 - size2)
                # if dist < 50 and (e1[1][0] + e1[1][1])/2 > 200 and (e1[1][0] + e1[1][1])/2 < 600 and abs(e1[1][0] - e1[1][1]) < 100:
                if (
                    dist < 50
                    and size1 > 80
                    and size1 < 400
                    and size_diff < 150
                    and el_var1 < 150
                    and el_var2 < 150
                ):
                    candidates.append((e1, e2, size1, size_diff, dist))

        for c in candidates:
            e1 = c[0]
            e2 = c[1]
            # print(c[2], c[3], c[4])
            # drawing the ellipses on the image
            cv2.ellipse(image, e1, (0, 255, 0), 2)
            cv2.ellipse(image, e2, (0, 255, 0), 2)

            size = (e1[1][0] + e1[1][1]) / 2

            print("size", str(size))
            center = (e1[0][1], e1[0][0])

            x1 = int(center[0] - size / 2)
            x2 = int(center[0] + size / 2)
            x_min = x1 if x1 > 0 else 0
            x_max = x2 if x2 < image.shape[0] else image.shape[0]

            y1 = int(center[1] - size / 2)
            y2 = int(center[1] + size / 2)
            y_min = y1 if y1 > 0 else 0
            y_max = y2 if y2 < image.shape[1] else image.shape[1]

            image_cp = image[x_min:x_max, y_min:y_max]
            print("dimensions", np.shape(image_cp))
            cv2.imshow("imag", image_cp)
            cv2.waitKey(1)

            image_cp = cv2.normalize(image_cp, image_cp, 0, 255, cv2.NORM_MINMAX)

            path = "./task3/food/image.jpg"
            cv2.imwrite(path, image_cp)

            food = self.recognize(path)
            print("FOOD", food)
            return food


if __name__ == "__main__":
    food_detector = FoodDetection()

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        food_detector.find_food()
        rate.sleep()

    cv2.destroyAllWindows()
