cmake_minimum_required(VERSION 2.8.3)
project(GLIO)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  message_generation
  pcl_ros
  pcl_conversions
  nlosExclusion
  gnss_comm
)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(GTSAM REQUIRED)

message(STATUS "Using catkin version ${catkin_VERSION}")
message(STATUS "Using Ceres version ${CERES_VERSION}")

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS roscpp message_runtime std_msgs nav_msgs geometry_msgs
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
)

link_directories(
    include
    ${GTSAM_LIBRARY_DIRS}
)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(Preprocessing src/Preprocessing.cpp)
target_link_libraries(Preprocessing ${catkin_LIBRARIES})

add_executable(LidarOdometry src/LidarOdometry.cpp)
target_link_libraries(LidarOdometry ${catkin_LIBRARIES} ${CERES_LIBRARIES})

add_executable(Estimator src/Estimator.cpp src/MarginalizationFactor.cpp)
target_link_libraries(Estimator ${catkin_LIBRARIES} ${CERES_LIBRARIES} gtsam pcl_visualization)

add_definitions(-w)