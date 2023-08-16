# 3DLA-GLIO

<p align='center'>
    <img src="./support_files/demo2cropcompressed.gif" alt="drawing" width="800"/>
</p>

**GLIO** is an accurate and robust online GNSS/LiDAR/IMU odometry system that tightly fuses the raw measurements from GNSS (pseudorange and Doppler), LiDAR, and IMU through non-linear factor graph optimization (FGO), which enables globally continuous and drift-free pose estimation even in highly challenging environments like urban canyons. 

The package is based on C++ which is compatible with the robot operation system (ROS) platform. Meanwhile, this package combines the RTKLIB (**[version: 2.4.3 b33](http://www.rtklib.com/)**) to read/decode the GNSS [RINEX](https://en.wikipedia.org/wiki/RINEX) files. Users from the Robotics field can easily have access to GNSS raw data for further study.

**Authors**: [Xikun Liu](https://www.polyu-ipn-lab.com/), [Weisong Wen](https://weisongwen.wixsite.com/weisongwen), [Li-ta Hsu](https://www.polyu-ipn-lab.com/) from the [Intelligent Positioning and Navigation Laboratory](https://www.polyu-ipn-lab.com/), The Hong Kong Polytechnic University. 

## System pipeline

<p align='center'>
    <img src="./support_files/pipeline.png" alt="drawing" width="600"/>
</p>

We propose a system that utilizes two stages of the optimization to achieve global consistent and continuous pose estimation in real-time.
  - In the first stage of optimization-the *optimizeSlidingWindow* stage, the sliding-window-based FGO is employed to integrate the GNSS-related factors, IMU pre-integration factor, and scan-to-map-based LiDAR factor for efficient odometry estimation.
  - In the second stage of optimization-the *optimizeBatch* stage, the LiDAR factor is employed as a scan-to-multiscan scheme to maintain global consistency and improve the robustness to the GNSS outlier by large-scale batch optimization.

**Video:**

[![GLIO Video](http://.jpg)](https://www.youtube.com/ "GLIO Video")


## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 18.04, ROS Melodic. [ROS Installation](http://wiki.ros.org/ROS/Installation). The package is tested on Ubuntu 18.04 with ROS Melodic. 

### 1.2. **Ceres Solver**
[Ceres Solver](https://ceres-solver.googlesource.com/ceres-solver) is used for optimization, please refer to the instructions on [GraphGNSSLib](https://github.com/weisongwen/GraphGNSSLib) to install Ceres-solver.

### 1.3. **Eigen**
[Eigen 3.3.3](https://gitlab.com/libeigen/eigen/-/archive/3.3.3/eigen-3.3.3.zip) is used for matrix calculation.

### 1.4. **Extra Libraries**
```bash
sudo apt-get install ros-melodic-novatel-msgs
```
### 1.5. **Pre-built Libraries**
[GraphGNSSLib V1.1](https://github.com/weisongwen/GraphGNSSLib.git) and [gnss_comm](https://github.com/HKUST-Aerial-Robotics/gnss_comm.git) is pre-built in the package.

## 2. **Build GLIO**
Clone the repository and catkin_make:
```bash
mkdir GLIO_ws/src
cd ~/GLIO_ws/src
mkdir result
git clone https://github.com/XikunLiu-huskit/3DLA-GLIO.git
cd ../
# if you fail in the last catkin_make, please source and catkin_make again
catkin_make
source ~/GLIO/devel/setup.bash
```
## 4. **Run GLIO with dataset *UrbanNav***
Launch GLIO via:
```
roslaunch GLIO run_urban_hk.launch
```
Open another terminal and launch the GNSS preprocessor by:
```
roslaunch global_fusion dataublox_Whampoa20210521.launch
```
Then play the bag:
```
rosbag play UrbanNav-HK_Whampoa-20210521_sensors.bag
```
Visit [UrbanNav](https://www.polyu-ipn-lab.com/download) and download more data sequences follow the [instruction](https://github.com/IPNL-POLYU/UrbanNavDataset/blob/master/docs/GETTING_STARTED.md).

## 6. **Acknowledgements**
GLIO is based on [LiLi-OM](https://github.com/KIT-ISAS/lili-om.git), [GraphGNSSLib](https://github.com/weisongwen/GraphGNSSLib.git), and [GVINS](https://github.com/HKUST-Aerial-Robotics/GVINS.git). The [rviz_satellite](https://github.com/nobleo/rviz_satellite) is used for visualization. Huge Thanks to their great work.

## 7. **Licence**
The source code is released under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html) license.
