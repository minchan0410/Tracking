cmake_minimum_required(VERSION 3.0.2)
project(fcos3d_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/fcos3d_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
