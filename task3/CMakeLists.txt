cmake_minimum_required(VERSION 3.0.2)
project(task3)

## Find system dependencies
find_package(Boost REQUIRED COMPONENTS system filesystem thread)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

## Poisci kitkat paketke :3 
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  dynamic_reconfigure
  genmsg
  geometry_msgs
  image_transport
  nodelet
  nodelet_topic_tools
  pcl_conversions
  message_generation
  pcl_msgs
  pluginlib
  rosbag
  rosconsole
  roscpp
  roslib
  rospy
  sensor_msgs
  std_msgs
  tf2_geometry_msgs
  tf2_ros
  tf
)


## Include directories
include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)


## Povezi direktorije
link_directories(
  ${Boost_LIBRARY_DIRS}
  ${Eigen_LIBRARY_DIRS}
  ${PCL_LIBRARY_DIRS}
  ${catkin_LIBRARY_DIRS}
)
  
link_libraries(${Boost_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_message_files(
  FILES
  ColorPose.msg
)

add_service_files(
	FILES
	speech.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

## Declare the kitkat pakidz
catkin_package(
  LIBRARIES
    pcl_ros_filters
    pcl_ros_io
    pcl_ros_tf
  CATKIN_DEPENDS 
    cv_bridge
    message_runtime
    dynamic_reconfigure
    image_transport
    pcl_conversions
    pcl_msgs
    roscpp
    sensor_msgs
    std_msgs
  DEPENDS
    Boost
    Eigen
    PCL
)


## Si plas plas 
add_executable (circle src/circle_detection.cpp)

add_dependencies(circle ${${PROJECT_NAME}_EXPORTED_TARGETS})

## Si plas plas links
target_link_libraries(circle ${catkin_LIBRARIES} ${OpenCV_INCLUDE_DIRS})

