#include "sensor_msgs/Image.h"
#include "task2/ColorPose.h"
#include <algorithm>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Pose.h>
#include <math.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/Marker.h>

// https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html

using namespace cv_bridge;
using namespace sensor_msgs;
using namespace message_filters;

typedef std::vector<cv::Vec3f> vec3arr;
typedef std::vector<cv::Vec4f> vec4arr;

typedef sync_policies::ApproximateTime<Image, Image> ApproxTimeSync;

ros::Publisher image_pub;
boost::shared_ptr<tf::TransformListener> listener;

ros::Publisher pubm;
ros::Publisher pos_publisher;

bool doDebug;
std::string camera_frame;

vec4arr circleDetect(cv::Mat input, cv::Mat output)
{
    vec4arr valid_circles, all_circles;

    // fixed arguments for HoughCircles
    int minRadius = 10;
    int maxRadius = 200;
    float accResolution = 2;
    int minDist = 100;

    // prepare arguments for circle detection
    int cannyTreshold = 100;
    int accTreshold = 75; // 75

    int centerTreshold = 50;

    // cv::blur(input, input, cv::Size(3, 3));

    cv::medianBlur(input, input, 5);

    // perform detection
    // all_circle = (x,y,radius,votes)[]
    cv::HoughCircles(input, all_circles, cv::HOUGH_GRADIENT, accResolution, minDist,
                     cannyTreshold, accTreshold, minRadius, maxRadius);

    // ROS_INFO("%d", all_circles.size());

    // ROS_INFO("----------------------------");

    // draw detected circles
    for (size_t i = 0; i < all_circles.size(); i++)
    {
        cv::Point center(cvRound(all_circles[i][0]), cvRound(all_circles[i][1]));

        // log circle to ROSINFo
        // if (doDebug)
        // {
        //     ROS_INFO("%ld: %d, %d, %d, %d", i, center.x, center.y, cvRound(all_circles[i][2]), (int)all_circles[i][3]);
        // }

        // only interested in hollow circles
        int center_color = input.at<uchar>(center);

        if (center_color > centerTreshold)
        {
            // ROS_INFO("Ignoring circle with center color: %d", center_color);
            continue;
        }

        // store valid circles for output
        valid_circles.insert(valid_circles.end(), all_circles[i]);

        int radius = cvRound(all_circles[i][2]);

        if (doDebug)
        {
            // yellow circle center
            cv::circle(output, center, 3, cv::Scalar(0, 255, 255), -1);

            // red circle outline
            cv::circle(output, center, radius, cv::Scalar(0, 0, 255), 1);
        }
    }

    return valid_circles;
}

void getDepths(vec4arr circles, const cv_bridge::CvImageConstPtr &depth_f, const cv_bridge::CvImageConstPtr &rgb_image, cv::Mat output, std_msgs::Header depth_header)
{
    // print the cv_ptr to console for debugging purposes
    // row by row, not commented

    // display the rgb_image to cv window
    // cv::imshow("rgb_image", rgb_image->image);
    // cv::waitKey(1);

    for (size_t i = 0; i < circles.size(); i++)
    {
        int min_x = std::max(cvRound(circles[i][0]) - cvRound(circles[i][2]), 0);
        int max_x = std::min(cvRound(circles[i][0]) + cvRound(circles[i][2]), depth_f->image.cols);
        int min_y = std::max(cvRound(circles[i][1]) - cvRound(circles[i][2]), 0);
        int max_y = std::min(cvRound(circles[i][1]) + cvRound(circles[i][2]), depth_f->image.rows);

        task2::ColorPose colorPose;

        colorPose.color.r = 0;
        colorPose.color.g = 0;
        colorPose.color.b = 0;

        cv::rectangle(output, cv::Point(min_x, min_y), cv::Point(max_x, max_y), cv::Scalar(255, 122, 0), 1);

        float accumulator = .0f;
        int count = 0;

        for (int y = min_y; y < max_y; y++)
        {
            for (int x = min_x; x < max_x; x++)
            {
                float depth = depth_f->image.at<float>(y, x);

                if (depth >= 0.1f)
                {
                    float distance = std::sqrt(std::pow(x - circles[i][0], 2) + std::pow(y - circles[i][1], 2));

                    if (distance <= circles[i][2])
                    {
                        accumulator += depth;
                        count++;

                        cv::Vec3b rgb = rgb_image->image.at<cv::Vec3b>(y, x);

                        // ROS_INFO("%d, %d, %d", rgb[0], rgb[1], rgb[2]);

                        colorPose.color.r += rgb[2];
                        colorPose.color.g += rgb[1];
                        colorPose.color.b += rgb[0];

                        output.at<cv::Vec3b>(y, x) = output.at<cv::Vec3b>(y, x) + cv::Vec3b(100, 100, 0);
                    }
                }
            }
        }

        if (count < 20)
            return;

        float distance = accumulator / count;

        colorPose.color.r /= count;
        colorPose.color.g /= count;
        colorPose.color.b /= count;

        // debug the color
        // ROS_INFO("aaaaaaaaaa %f, %f, %f", colorPose.color.r, colorPose.color.g, colorPose.color.b);

        // ROS_DEBUG("Depth is %f", distance);

        float k_f = 554;

        double angle_to_target = atan2(depth_f->image.cols / 2 - circles[i][0], k_f);

        float x_t = distance * std::cos(angle_to_target);
        float y_t = distance * std::sin(angle_to_target);

        //  define  a new point stamped
        geometry_msgs::PointStamped point_stamped;

        point_stamped.point.x = -y_t;
        point_stamped.point.y = 0;
        point_stamped.point.z = x_t;
        point_stamped.header.frame_id = camera_frame;
        point_stamped.header.stamp = depth_header.stamp;

        // print point stamped x y z
        // ROS_INFO("x: %f, y: %f, z: %f", point_stamped.point.x, point_stamped.point.y, point_stamped.point.z);

        // create a new pose
        geometry_msgs::Pose pose;

        try
        {
            listener->transformPoint("map", point_stamped, point_stamped);

            pose.position.x = point_stamped.point.x;
            pose.position.y = point_stamped.point.y;
            pose.position.z = point_stamped.point.z;

            colorPose.pose = pose;

            pos_publisher.publish(colorPose);

            if (doDebug)
            {

                visualization_msgs::Marker marker;

                marker.header.frame_id = "map";
                marker.header.stamp = ros::Time::now();

                marker.ns = "cylinder";
                marker.id = 0;

                marker.type = visualization_msgs::Marker::CUBE;
                marker.action = visualization_msgs::Marker::ADD;

                marker.pose.position.x = pose.position.x;
                marker.pose.position.y = pose.position.y;
                marker.pose.position.z = pose.position.z;
                marker.pose.orientation.w = 1.0;

                marker.scale.x = 0.1;
                marker.scale.y = 0.1;
                marker.scale.z = 0.1;

                marker.color.r = 0.0f;
                marker.color.g = 1.0f;
                marker.color.b = 0.0f;
                marker.color.a = 1.0f;

                marker.lifetime = ros::Duration();

                pubm.publish(marker);
            }
            // ROS_DEBUG("x: %f, y: %f, z: %f", pose.position.x, pose.position.y, pose.position.z);
        }
        catch (tf::TransformException ex)
        {
            continue;
        }
    }
}

void handleImage(const sensor_msgs::ImageConstPtr &depth_image, const sensor_msgs::ImageConstPtr &rgb_image)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    cv_bridge::CvImageConstPtr cv_rgb;

    try
    {
        cv_ptr = cv_bridge::toCvCopy(depth_image);
        cv_rgb = cv_bridge::toCvCopy(rgb_image);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // message contaings 32-bit floating point depth image
    // convert to standard 8-bit gray scale image
    cv::Mat mono8_img = cv::Mat(cv_ptr->image.size(), CV_8UC1);
    cv::convertScaleAbs(cv_ptr->image, mono8_img, 100, 0.0);

    // create separate rgb image for displaying results
    cv::Mat rgb_img = cv::Mat(mono8_img.size(), CV_8UC3);
    cv::cvtColor(mono8_img, rgb_img, cv::COLOR_GRAY2RGB);

    // detect circles in the depth map
    vec4arr circles = circleDetect(mono8_img, rgb_img);

    getDepths(circles, cv_ptr, cv_rgb, rgb_img, depth_image->header);

    // publish image with detections

    if (doDebug)
    {
        cv_bridge::CvImage out_msg;

        out_msg.header = depth_image->header; // Same timestamp and tf frame as input image
        out_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
        out_msg.image = rgb_img;

        image_pub.publish(out_msg.toImageMsg());
    }
}

int main(int argc, char **argv)
{
    // Ros initialization and parameter harvesting

    ros::init(argc, argv, "ring_detect");
    ros::NodeHandle nh("~");

    std::string depth_topic, rgb_topic, ring_marker_topic, ring_image_topic, hits;

    bool param_debug = false;

    nh.getParam("depth", depth_topic);
    nh.getParam("rgb", rgb_topic);
    nh.getParam("ring_marker", ring_marker_topic);
    nh.getParam("ring_image", ring_image_topic);
    nh.getParam("hits", hits);
    nh.getParam("debug", param_debug);
    nh.getParam("camera_frame", camera_frame);

    doDebug = param_debug;

    if (depth_topic.empty() || rgb_topic.empty() || ring_marker_topic.empty() || ring_image_topic.empty() || hits.empty() || camera_frame.empty())

    {
        ROS_ERROR("One or more of the parameters is not set. Please check the launch file.");
        return -1;
    }

    listener.reset(new tf::TransformListener());

    ROS_INFO("Circle detection started");
    ROS_INFO("Depth topic: %s", depth_topic.c_str());
    ROS_INFO("RGB topic: %s", rgb_topic.c_str());
    ROS_INFO("Ring marker topic: %s", ring_marker_topic.c_str());
    ROS_INFO("Ring image topic: %s", ring_image_topic.c_str());
    ROS_INFO("Hits topic: %s", hits.c_str());
    ROS_INFO("Debug: %s", param_debug ? "true" : "false");

    // Subcribtions and synchronizers
    Subscriber<Image> depth_sub(nh, depth_topic, 1);
    Subscriber<Image> rgb_sub(nh, rgb_topic, 1);

    Synchronizer<ApproxTimeSync> sync(ApproxTimeSync(5), depth_sub, rgb_sub);

    sync.registerCallback(boost::bind(&handleImage, _1, _2));

    if (doDebug)
    {
        image_pub = nh.advertise<Image>(ring_image_topic, 1);
        pubm = nh.advertise<visualization_msgs::Marker>(ring_marker_topic, 1);
    }

    pos_publisher = nh.advertise<task2::ColorPose>(hits, 1000);

    ros::spin();

    return 0;
}