cmake_minimum_required(VERSION 2.8.3)
project(momo)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)



catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES momo
#  CATKIN_DEPENDS roscpp rospy
#  DEPENDS system_lib
)


# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)
