cmake_minimum_required(VERSION 3.0.2)
project(task1)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	cv_bridge
	sensor_msgs
	nav_msgs
	message_generation
	std_msgs
	geometry_msgs
	tf2
	tf2_geometry_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
   FILES
   FaceLocation.msg
)

add_service_files(
	FILES
	Normal_vector.srv
)

generate_messages(
	DEPENDENCIES
	std_msgs
)

find_package(OpenCV)

catkin_package()

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})