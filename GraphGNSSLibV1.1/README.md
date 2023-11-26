# GraphGNSSLib
### An Open-source Package for GNSS Positioning and Real-time Kinematic Using Factor Graph Optimization

This repository is the implementation of the open-sourced package, the GraphGNSSLib, which makes use of the factor graph optimization (FGO) to perform the GNSS positioning and real-time kinematic (RTK) positioning. In this package, measurements from the historical and current epochs are structured into a factor graph which is then solved by non-linear optimization. The package is based on C++ which is compatible with the robot operation system (ROS) platform. Meanwhile, this package combines the RTKLIB (**[version: 2.4.3 b33](http://www.rtklib.com/)**) to read/decode the GNSS [RINEX](https://en.wikipedia.org/wiki/RINEX) files. Users from Robotics field can easily have access to GNSS raw data for further study.(**update date: 2020/11/30**)

**Important Notes**: 
  - Be noted that the **GNSS Positioning** mentioned throughout the package means estimating the positioing of the GNSS receiver based on the combination of pseudorange and Doppler measurements uisng FGO.
  - Be noted that the **GNSS-RTK Positioning** mentioned throughout the package means estimating the positioing (float solution) of the GNSS receiver based on the combination of double-differenced pseudorange, carrier-phase and the Doppler measurements using FGO. Finally, the ambiguity is resolved using LAMBDA algorithm.

**Authors**: [Weisong Wen](https://weisongwen.wixsite.com/weisongwen), [Li-ta Hsu](https://www.polyu-ipn-lab.com/) from the [Intelligent Positioning and Navigation Laboratory](https://www.polyu-ipn-lab.com/), The Hong Kong Polytechnic University

**Related Papers:** (paper is not exactly same with code)
  - Wen Weisong., Hsu, Li-Ta.* (2020) **GraphGNSSLib: An Open-source Package for GNSS Positioning and Real-time Kinematic Using Factor Graph Optimization**, GPS Solutions. (**Submitted**)

*if you use GraphGNSSLib for your academic research, please cite our related [papers](https://www.polyu-ipn-lab.com/)*

<p align="center">
  <img width="712pix" src="img/software_flowchart.png">
</p>

<center> Software flowchart of GraphGNSSLib, more information please refer to mannual and paper.</center>

## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 16.04, ROS Kinetic. [ROS Installation](http://wiki.ros.org/ROS/Installation). We only test it on Ubuntu 16.04 with ROS Kinetic. 

### 1.2. **Ceres Solver**
Follow the following instructions to install Ceres-solver instead of using the latest version of Ceres-solver.

**Step 1**: Download the [Ceres-solver](https://github.com/weisongwen/GraphGNSSLib/tree/master/support_files) which is compatible with GraphGNSSLib. 

**Step 2**: make and install
```bash
sudo apt-get install cmake
# google-glog + gflags
sudo apt-get install libgoogle-glog-dev
# BLAS & LAPACK
sudo apt-get install libatlas-base-dev
# Eigen3
sudo apt-get install libeigen3-dev
# make Ceres-solver
mkdir ceres-bin
cd ceres-bin
cmake ../ceres-solver
sudo make -j4
sudo make test
sudo make install
```

### 1.3. **Extra Libraries**
```bash
sudo apt-get install ros-kinetic-novatel-msgs
```
## 2. Build GraphGNSSLib
Clone the repository and catkin_make:
```bash
mkdir GraphGNSSLib/src
cd ~/GraphGNSSLib/src
mkdir result
git clone https://github.com/weisongwen/GraphGNSSLib.git
cd ../
# if you fail in the last catkin_make, please source and catkin_make again
catkin_make
source ~/GraphGNSSLib/devel/setup.bash
catkin_make
```
(**if you fail in this step, try to find another computer with clean system or reinstall Ubuntu and ROS**)

## 3. Run GNSS positioning via FGO using dataset [UrbanNav](https://www.polyu-ipn-lab.com/download)   
The GNSS positioning via FGO is validated using static dataset collected near TST of Hong Kong. Several parameters are as follows:
  - GPS second span: **46701** to **47185**
  - satellite system: **GPS/BeiDou**
  - Window Size: **Batch**
  - measurements considered: double-differenced pseudorange and carrier-phase measurements, Doppler measurements
  - result is saved by default
    ```c++
    ~/GraphGNSSLib/trajectory_psr_dop_fusion.csv
    ```

please enable the following in rtklib.h
```bash
#define RTK_FGO 0
```
- Solution 1 to run the GNSS positioning Demo
  ```bash
  source ~/GraphGNSSLib/devel/setup.bash
  # read GNSS raw data and publish as ROS topic
  # we provide several datasets, enjoy it!
  roslaunch global_fusion dataublox_TST20190428.launch
  # run pseudorange and doppler fusion
  roslaunch global_fusion psr_doppler_fusion.launch
  ```
<p align="center">
  <img width="712pix" src="img/SPP_trajectory1.png">
</p>
<center> Trajectories of three methods (GNSS positioning using WLS with the red curve, GNSS positioning using EKF with the green curve, and GNSS positioning using FGO with blue curve throughout the test. The x-axis and y-axis denote the east and north directions, respectively</center>


## 4. Run GNSS RTK-FGO using static dataset   
The GNSS RTK-FGO is validated using static dataset collected near TST of Hong Kong. Several parameters are as follows:
  - GPS second span: **270149** to **270306**
  - satellite system: **GPS/BeiDou**
  - Window Size: **Batch**
  - measurements considered: double-differenced pseudorange and carrier-phase measurements, Doppler measurements
  - result is saved by default
    ```c++
    ~/GraphGNSSLib/FGO_trajectoryllh_pdrtk.csv
    ```

please enable the following in rtklib.h
```bash
#define RTK_FGO 1
```
- Solution 1 to run the RTK-FGO Demo
  ```bash
  source ~/GraphGNSSLib/devel/setup.bash
  # read GNSS raw data and publish as ROS topic
  roslaunch global_fusion dataublox_TST20200603.launch
  # run GNSS RTK
  roslaunch global_fusion psr_doppler_car_rtk.launch
  ```
<p align="center">
  <img width="712pix" src="img/RTK_trajectory.png">
</p>
<center> Trajectories of three methods (RTK-EKF with the red dots and RTK-FGO with the blue dots throughout the test. The x-axis and y-axis denote the east and north directions, respectively.</center>


## 5. Additional Functions
- [RTKLIB test](global_fusion/src/testRTKLIBNode.cpp)
  - ```rosrun global_fusion testRTKLIBNode```  
- [GNSS single point positioning using Ceres Solver AutoDiffCostFunction](global_fusion/src/gnss_estimator/psr_spp.cpp)
  - ```roslaunch global_fusion psr_spp.launch```
- [GNSS single point positioning using Ceres Solver DynamicAutoDiffCostFunction](global_fusion/src/gnss_estimator/psr_spp_dyna_auto.cpp)
  - ```roslaunch global_fusion psr_spp.launch```
- [GNSS/Doppler fusion using EKF](global_fusion/src/gnss_estimator/psr_doppler_fusion_EKF.cpp)
  - ```roslaunch global_fusion psr_doppler_fusion_EKF.launch```
- [pseudorange/doppler/carrier-phase RTK using Ceres Solver DynamicAutoDiffCostFunction](global_fusion/src/gnss_estimator/psr_doppler_fusion_EKF.cpp)
  - ```roslaunch global_fusion psr_doppler_car_rtk_dyna.launch```
- [GNSS/INS loosely integration using EKF](global_fusion/src/gnss_ins_estimator/gnss_ins_lc_ekf_fusion.cpp)
  - ```roslaunch global_fusion lc_gnss_ins.launch```

- [GNSS/Doppler fusion using FGO-GNC](global_fusion/src/gnss_estimator_robust/psr_doppler_fusion_gnc_gm.cpp)
  - ```roslaunch global_fusion psr_doppler_fusion_gnc_gm.launch```
- [evaluate the accuracy](global_fusion/src/evo/evaluate_gps_solution.cpp)
  - ```roslaunch global_fusion evo.launch```
- [gnss ins integration using tc with ins provide enu](global_fusion/src/gnss_ins_estimator/gnss_insenu_tc_ekf_fusion.cpp)
  - ```roslaunch global_fusion lc_gnss_insenu.launch```
  - The GNSS data is from rinex files (GEOP318C.20o from huawei phone)
  - Satellite systems include the GPS (hksc3180.20n), GLONASS (hksc3180.20g), BeiDou (hksc3180.20b)
  - The ground truth is from file Seaside2TSTGT.csv

- [pseudorange/doppler/carrier-phase integration free of ambiguity](global_fusion/src/gnss_estimator/psr_car_doppler_fusion.cpp)
  - Info: this is the code for the paper sliding window carrier-phase aided GNSS positioning
  - ```roslaunch global_fusion dataublox_KLT20200314.launch```
  - ```roslaunch global_fusion psr_car_doppler_fusion.launch```
  - ```roslaunch global_fusion evo_KLT_ublox_M8T_0314.launch```
  - several key points:
    - ```residuals[k] = T(1) * residuals[k] / (0.05*sqrt(covMat(k,k)) );```
    - ```c++
      loss_function_Carrier = new ceres::CauchyLoss(1.0);
            CCCConstraint::CCCCostFunction* cost_function =
            CCCConstraint::Create(SCCC, &state_array,&parameter_blocks);
      problem.AddResidualBlock(cost_function, loss_function_Carrier, parameter_blocks);
      ```
    - ```c++
      if((m==(length-1)) || (SCCC.length>15)) 
      {
          // existFlags[m]==false;
          findEnd = true;
          // std::cout << "SCCC.length:" <<SCCC.length << std::endl;
      }
      ```
  - The GNSS data is from rinex files (GEOP318C.20o from huawei phone)
  - Satellite systems include the GPS (hksc3180.20n), GLONASS (hksc3180.20g), BeiDou (hksc3180.20b)
  - The ground truth is from file Seaside2TSTGT.csv


<details>
<summary>rosbag generator for further data processing</summary>

- [rosbag generator](global_fusion/src/gnss_preprocessor/rosbag_generator.cpp): add the GNSS raw data (pseudorange, Carrier-phase adn Doopler velocity to the existing rosbag). This is to be opensourced in the coming future.
  - Step 1: read the rinex files, the rosbag_generator_node accumulate the GNSS data.
    - ```roslaunch global_fusion rosbag_generator.launch``` and 
    ```roslaunch global_fusion dataublox_MDD20200314.launch```
  - Step 2: replay the rosbag with span-cpt data, re-record the rosbag. Currently, several rosbag data are available for further evaluation (Data: GNSS raw/INS/LiDAR/Camera/SPAN-CPT for 3DLA GNSS-RTK, etc).
    - Dataset 1 (TST 20190428 dataset): **2019-04-28-20-58-02.bag** **---->** **2019-04-28-20-58-02_addRawGNSS.bag**; 
      ```bash
      path:        2019-04-28-20-58-02_addRawGNSS.bag
      version:     2.0
      duration:    8:07s (487s)
      start:       Jan 01 2021 14:03:23.13 (1609481003.13)
      end:         Jan 01 2021 14:11:30.20 (1609481490.20)
      size:        40.4 GB
      messages:    68194
      compression: none [9737/9737 chunks]
      types:       geometry_msgs/Pose           [e45d45a5a1ce597b249e23fb30fc871f]
                  nav_msgs/Odometry            [cd5e73d190d741a2f92e81eda573aca7]
                  nlosExclusion/GNSS_Raw_Array [6a69c5dd28d3ae527f3fdefebda5da69]
                  novatel_msgs/BESTPOS         [8321b9523105411643891c8653878967]
                  novatel_msgs/CORRIMUDATA     [8ca3f26f898322425170fe621393f009]
                  novatel_msgs/INSCOV          [75d77cf9321af3888caeeab3a756d0ac]
                  novatel_msgs/INSPVAX         [b5d66747957184042a6cca9b7368742f]
                  sensor_msgs/Image            [060021388200f6f0f447d0fcd9c64743]
                  sensor_msgs/Imu              [6a62c6daae103f4ff57a132d6f95cec2]
                  sensor_msgs/NavSatFix        [2d3a8cd499b9b4a0249fb98fd05cfa48]
                  sensor_msgs/PointCloud2      [1158d486dd51d683ce2f1be655c3c181]
                  ublox_msgs/AidALM            [de5ab2550e698fc8acfb7263c7c55fa2]
                  ublox_msgs/AidEPH            [796d86b27ebfe497b3a42695f2e69e13]
                  ublox_msgs/MonHW             [605e9f0118e26136185358e2b10a0913]
                  ublox_msgs/NavCLOCK          [a9acfdf2e7ac2bf086926ae4e6a182a0]
                  ublox_msgs/NavPVT            [10f57b0db1fa3679c06567492fa4e5f2]
                  ublox_msgs/NavSAT            [e8ea6afd23cb79e7e7385313416d9c15]
                  ublox_msgs/NavSTATUS         [68047fb8ca04a038a6b031cd1a908762]
                  ublox_msgs/RxmRAWX           [a2df4b27b6a2a1565e42f5669dbb11b5]
      topics:      /camera/image_color                          4870 msgs    : sensor_msgs/Image           
                  /imu/data                                   48707 msgs    : sensor_msgs/Imu             
                  /navsat/fix                                   487 msgs    : sensor_msgs/NavSatFix       
                  /navsat/odom                                  487 msgs    : nav_msgs/Odometry           
                  /navsat/origin                                  1 msg     : geometry_msgs/Pose          
                  /novatel_data/bestpos                         487 msgs    : novatel_msgs/BESTPOS        
                  /novatel_data/corrimudata                     487 msgs    : novatel_msgs/CORRIMUDATA    
                  /novatel_data/inscov                          487 msgs    : novatel_msgs/INSCOV         
                  /novatel_data/inspvax                         487 msgs    : novatel_msgs/INSPVAX        
                  /rosbag_generator_node/GNSSDopVelRov1         482 msgs    : nav_msgs/Odometry           
                  /rosbag_generator_node/GNSSPsrCarRov1         482 msgs    : nlosExclusion/GNSS_Raw_Array
                  /rosbag_generator_node/GNSSPsrCarStation1     482 msgs    : nlosExclusion/GNSS_Raw_Array
                  /ublox_node/aidalm                            487 msgs    : ublox_msgs/AidALM           
                  /ublox_node/aideph                            487 msgs    : ublox_msgs/AidEPH           
                  /ublox_node/fix                               487 msgs    : sensor_msgs/NavSatFix       
                  /ublox_node/monhw                             487 msgs    : ublox_msgs/MonHW            
                  /ublox_node/navclock                          487 msgs    : ublox_msgs/NavCLOCK         
                  /ublox_node/navpvt                            487 msgs    : ublox_msgs/NavPVT           
                  /ublox_node/navsat                             24 msgs    : ublox_msgs/NavSAT           
                  /ublox_node/navstatus                         487 msgs    : ublox_msgs/NavSTATUS        
                  /ublox_node/rxmraw                           1949 msgs    : ublox_msgs/RxmRAWX          
                  /velodyne_points                             4866 msgs    : sensor_msgs/PointCloud2
      ```
    - Dataset 2 (KLT 20200314 dataset): **2020-03-14-16-45-35.bag** **---->** **2020-03-14-16-45-35_addRawGNSS.bag**; 
      ```bash
      path:        2020-03-14-16-45-35_addRawGNSS.bag
      version:     2.0
      duration:    5:01s (301s)
      start:       Jan 01 2021 15:15:25.59 (1609485325.59)
      end:         Jan 01 2021 15:20:27.49 (1609485627.49)
      size:        25.1 GB
      messages:    68889
      compression: none [6011/6011 chunks]
      types:       geometry_msgs/Pose           [e45d45a5a1ce597b249e23fb30fc871f]
                  nav_msgs/Odometry            [cd5e73d190d741a2f92e81eda573aca7]
                  nlosExclusion/GNSS_Raw_Array [6a69c5dd28d3ae527f3fdefebda5da69]
                  novatel_msgs/BESTPOS         [8321b9523105411643891c8653878967]
                  novatel_msgs/CORRIMUDATA     [8ca3f26f898322425170fe621393f009]
                  novatel_msgs/INSCOV          [75d77cf9321af3888caeeab3a756d0ac]
                  novatel_msgs/INSPVAX         [b5d66747957184042a6cca9b7368742f]
                  rosgraph_msgs/Log            [acffd30cd6b6de30f120938c17c593fb]
                  sensor_msgs/Image            [060021388200f6f0f447d0fcd9c64743]
                  sensor_msgs/Imu              [6a62c6daae103f4ff57a132d6f95cec2]
                  sensor_msgs/NavSatFix        [2d3a8cd499b9b4a0249fb98fd05cfa48]
                  sensor_msgs/PointCloud2      [1158d486dd51d683ce2f1be655c3c181]
      topics:      /camera/image_color                          3005 msgs    : sensor_msgs/Image           
                  /imu/data                                   60147 msgs    : sensor_msgs/Imu             
                  /navsat/fix                                   301 msgs    : sensor_msgs/NavSatFix       
                  /navsat/odom                                  300 msgs    : nav_msgs/Odometry           
                  /navsat/origin                                  1 msg     : geometry_msgs/Pose          
                  /novatel_data/bestpos                         301 msgs    : novatel_msgs/BESTPOS        
                  /novatel_data/corrimudata                     301 msgs    : novatel_msgs/CORRIMUDATA    
                  /novatel_data/inscov                          300 msgs    : novatel_msgs/INSCOV         
                  /novatel_data/inspvax                         300 msgs    : novatel_msgs/INSPVAX        
                  /rosbag_generator_node/GNSSDopVelRov1         300 msgs    : nav_msgs/Odometry           
                  /rosbag_generator_node/GNSSPsrCarRov1         300 msgs    : nlosExclusion/GNSS_Raw_Array
                  /rosbag_generator_node/GNSSPsrCarStation1     300 msgs    : nlosExclusion/GNSS_Raw_Array
                  /rosout                                        16 msgs    : rosgraph_msgs/Log            (3 connections)
                  /rosout_agg                                    12 msgs    : rosgraph_msgs/Log           
                  /velodyne_points                             3005 msgs    : sensor_msgs/PointCloud2
      ```
    - Dataset 3 (MDD 20200314 dataset): **2020-03-14-18-15-17.bag** **---->** **2020-03-14-18-15-17_addRawGNSS.bag**; 
      ```bash
      path:        2020-03-14-18-15-17_addRawGNSS.bag
      version:     2.0
      duration:    26:39s (1599s)
      start:       Jan 01 2021 15:44:01.59 (1609487041.59)
      end:         Jan 01 2021 16:10:41.15 (1609488641.15)
      size:        133.0 GB
      messages:    364996
      compression: none [31969/31969 chunks]
      types:       geometry_msgs/Pose           [e45d45a5a1ce597b249e23fb30fc871f]
                  nav_msgs/Odometry            [cd5e73d190d741a2f92e81eda573aca7]
                  nlosExclusion/GNSS_Raw_Array [6a69c5dd28d3ae527f3fdefebda5da69]
                  novatel_msgs/BESTPOS         [8321b9523105411643891c8653878967]
                  novatel_msgs/CORRIMUDATA     [8ca3f26f898322425170fe621393f009]
                  novatel_msgs/INSCOV          [75d77cf9321af3888caeeab3a756d0ac]
                  novatel_msgs/INSPVAX         [b5d66747957184042a6cca9b7368742f]
                  rosgraph_msgs/Log            [acffd30cd6b6de30f120938c17c593fb]
                  sensor_msgs/Image            [060021388200f6f0f447d0fcd9c64743]
                  sensor_msgs/Imu              [6a62c6daae103f4ff57a132d6f95cec2]
                  sensor_msgs/NavSatFix        [2d3a8cd499b9b4a0249fb98fd05cfa48]
                  sensor_msgs/PointCloud2      [1158d486dd51d683ce2f1be655c3c181]
      topics:      /camera/image_color                          15989 msgs    : sensor_msgs/Image           
                  /imu/data                                   319872 msgs    : sensor_msgs/Imu             
                  /navsat/fix                                   1600 msgs    : sensor_msgs/NavSatFix       
                  /navsat/odom                                  1599 msgs    : nav_msgs/Odometry           
                  /navsat/origin                                   1 msg     : geometry_msgs/Pose          
                  /novatel_data/bestpos                         1600 msgs    : novatel_msgs/BESTPOS        
                  /novatel_data/corrimudata                     1600 msgs    : novatel_msgs/CORRIMUDATA    
                  /novatel_data/inscov                          1599 msgs    : novatel_msgs/INSCOV         
                  /novatel_data/inspvax                         1599 msgs    : novatel_msgs/INSPVAX        
                  /rosbag_generator_node/GNSSDopVelRov1         1181 msgs    : nav_msgs/Odometry           
                  /rosbag_generator_node/GNSSPsrCarRov1         1181 msgs    : nlosExclusion/GNSS_Raw_Array
                  /rosbag_generator_node/GNSSPsrCarStation1     1181 msgs    : nlosExclusion/GNSS_Raw_Array
                  /rosout                                         15 msgs    : rosgraph_msgs/Log            (3 connections)
                  /velodyne_points                             15979 msgs    : sensor_msgs/PointCloud2
      ```
</details>

## 6. Acknowledgements
We use [Ceres-solver](http://ceres-solver.org/) for non-linear optimization and [RTKLIB](http://www.rtklib.com/) for GNSS data decoding, etc. If there is any thing inappropriate, please contact me through 17902061r@connect.polyu.hk ([Weisong WEN](https://weisongwen.wixsite.com/weisongwen)).

## 7. License
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license. We are still working on improving the code reliability. For any technical issues, please contact Weisong Wen <17902061r@connect.polyu.hk>. For commercial inquiries, please contact Li-ta Hsu <lt.hsu@polyu.edu.hk>.

## 8. Related Publication

1. Wen, Weisong, Guohao Zhang, and Li-Ta Hsu. "Exclusion of GNSS NLOS receptions caused by dynamic objects in heavy traffic urban scenarios using real-time 3D point cloud: An approach without 3D maps." Position, Location and Navigation Symposium (PLANS), 2018 IEEE/ION. IEEE, 2018. (https://ieeexplore.ieee.org/abstract/document/8373377/)

2. Wen, W.; Hsu, L.-T.*; Zhang, G. (2018) Performance analysis of NDT-based graph slam for autonomous vehicle in diverse typical driving scenarios of Hong Kong. Sensors 18, 3928.

3. Wen, W., Zhang, G., Hsu, Li-Ta (Presenter), Correcting GNSS NLOS by 3D LiDAR and Building Height, ION GNSS+, 2018, Miami, Florida, USA.

4. Zhang, G., Wen, W., Hsu, Li-Ta, Collaborative GNSS Positioning with the Aids of 3D City Models, ION GNSS+, 2018, Miami, Florida, USA. (Best Student Paper Award)

5. Zhang, G., Wen, W., Hsu, Li-Ta, A Novel GNSS based V2V Cooperative Localization to Exclude Multipath Effect using Consistency Checks, IEEE PLANS, 2018, Monterey, California, USA.
Copyright (c) 2018 Weisong WEN

6. Wen Weisong., Tim Pfeifer., Xiwei Bai., Hsu, L.T.* Comparison of Extended Kalman Filter and Factor Graph Optimization for GNSS/INS Integrated Navigation System, The Journal of Navigation, 2020, (SCI. 2019 IF. 3.019, Ranking 10.7%) [Submitted]