/*******************************************************
 * Copyright (C) 2023, Intelligent Positioning and Navigation Lab, Hong Kong Polytechnic University
 * 
 * This file is part of GLIO.
 * Author: Xikun Liu (xi-kun.liu@connect.polyu.hk)
 * 
 * We make reference to lili-om, GVINS, etc., and we fully respect these works. Also thanks to the RTKLIB, GraphGNSSLib, gnss_common, etc.
 * Li, K., Li, M., & Hanebeck, U. D. (2021). Towards high-performance solid-state-lidar-inertial odometry and mapping. IEEE Robotics and Automation Letters, 6(3), 5167-5174.
 * Cao, S., Lu, X., & Shen, S. (2022). GVINS: Tightly coupled GNSS–visual–inertial fusion for smooth and consistent state estimation. IEEE Transactions on Robotics, 38(4), 2004-2021.
 * Wen, W., & Hsu, L. T. (2021, May). Towards robust GNSS positioning and real-time kinematic using factor graph optimization. In 2021 IEEE International Conference on Robotics and Automation (ICRA) (pp. 5884-5890). IEEE.
 * 
 * Liu, X., Wen, W., & Hsu, L. T. (2023). GLIO: Tightly-Coupled GNSS/LiDAR/IMU Integration for Continuous and Drift-free State Estimation of Intelligent Vehicles in Urban Areas. IEEE Transactions on Intelligent Vehicles.

 * Date: 2023/08/01
 *******************************************************/

#include "utils/common.h"
#include "utils/math_tools.h"
#include "utils/timer.h"
#include "utils/random_generator.hpp"
#include "factors/LidarKeyframeFactor.h"
#include "factors/LidarPoseFactor.h"
#include "factors/ImuFactor.h"
#include "factors/PriorFactor.h"
#include "factors/Preintegration.h"
#include "factors/MarginalizationFactor.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/navigation/GPSFactor.h>

// GNSS related header files 
#include <nlosExclusion/GNSS_Raw_Array.h>
#include <nlosExclusion/GNSS_Raw.h>
#include <nlosExclusion/GNSS_Raw_mf.h>
#include "utils/gnss_tools.h"

#include <gnss_comm/gnss_ros.hpp>
#include <gnss_comm/gnss_utility.hpp>
#include <gnss_comm/GnssPVTSolnMsg.h>

// GNSS Doppler factor and DD pseudorange factor
#include "factors/dopp_factor.hpp"
#include "factors/dd_psr_factor.hpp"

#include <sensor_msgs/NavSatFix.h>
#include <std_msgs/Float32MultiArray.h>

#include <novatel_msgs/INSPVAX.h> // novatel_msgs/INSPVAX
#include <novatel_msgs/BESTPOS.h> // novatel_msgs/INSPVAX

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>

#include <random>

/* tools used for gnss related calculation */
using namespace gnss_comm;
using namespace gtsam;

#define EPOCH_SIZE 5000
#define lossKernel 1.0
#define Doppler2PSRWeight 0.1 // original is 10 5

class Estimator {
private:
    ros::NodeHandle nh; // node handle

    ros::Subscriber sub_surf; // sub surface features
    ros::Subscriber sub_odom; // sub odom from scan-to-scan matching
    ros::Subscriber sub_each_odom; // sub relative motion between two scan
    ros::Subscriber sub_full_cloud; // sub full clouds from lidar odomtry node
    ros::Subscriber sub_imu; // sub imu data

    ros::Publisher pub_map; // publish map
    ros::Publisher pub_odom; // pub odom mapped
    ros::Publisher pub_batch_path; // pub batch traj
    ros::Publisher pub_odom_rtk_ini, pub_path_rtk_ini; // pub rtklib traj

    ros::Publisher pub_gnss_lla;
    ros::Publisher pub_tc_enu_path, pub_lc_enu_path;
    ros::Publisher pub_poses; // pub trajectories in pcl format
    ros::Publisher pub_surf; //publish surface map
    ros::Publisher pub_full; // publish raw scan with transformed pose from tc fusion
    ros::Publisher span_BP_pub; //publish gt data by txt reading

    ros::Publisher pub_sub_gl_map; // publish sub map

    nav_msgs::Path rtk_ini_enu_path, tc_enu_path, spp_path;
    nav_msgs::Odometry odom_mapping; // odom from tc fusion and lc fusion
    nav_msgs::Odometry odom_init_kf; //
    std_msgs::Float32MultiArray rtk_infos;

    /* information flag from lidar odometry */
    bool new_surf = false;
    bool new_odom = false;
    bool new_each_odom = false;
    bool new_full_cloud = false;

    /* time for the information from lidar odometry */
    double time_new_odom;

    /* batch searching range */
    int search_range = 6;
    int batch_activate_idx = 0;

    /* parameter for lidar feature selection */
    int feature_res_num;
    int rand_set_num;
    int batch_feature_res_num;
    int batch_rand_set_num;
    bool random_select;

    pcl::PointCloud<PointType>::Ptr surf_last;
    pcl::PointCloud<PointType>::Ptr full_cloud;
    vector<pcl::PointCloud<PointType>::Ptr> full_clouds_ds;

    pcl::PointCloud<PointType>::Ptr surf_last_ds;

    vector<pcl::PointCloud<PointType>::Ptr> surf_lasts_ds;

    pcl::PointCloud<PointType>::Ptr surf_local_map;
    pcl::PointCloud<PointType>::Ptr surf_global_map;

    pcl::PointCloud<PointType>::Ptr surf_local_map_ds;
    pcl::PointCloud<PointType>::Ptr surf_global_map_ds; // downsampled

    vector<pcl::PointCloud<PointType>::Ptr> vec_surf_cur_pts; // save points
    vector<pcl::PointCloud<PointType>::Ptr> vec_surf_normal; // save normals
    vector<vector<double>> vec_surf_scores; // constant * weights

    map<int, map<int, pcl::PointCloud<PointType>::Ptr>> gl_vec_surf_cur_pts; // save points
    map<int, map<int, pcl::PointCloud<PointType>::Ptr>> gl_vec_surf_cur_pts_startend; // save points

    map<int, map<int, vector<double>>> gl_vec_surf_scores; // constant * weights
    map<int, map<int, vector<double>>> gl_vec_surf_scores_startend; // constant * weights

    map<int, map<int, vector<vector<double>>>> gl_vec_surf_normals_cents;
    map<int, map<int, vector<vector<double>>>> gl_vec_surf_normals_cents_startend;

    pcl::PointCloud<PointType>::Ptr latest_key_frames;
    pcl::PointCloud<PointType>::Ptr latest_key_frames_ds;
    pcl::PointCloud<PointType>::Ptr his_key_frames;
    pcl::PointCloud<PointType>::Ptr his_key_frames_ds;

    pcl::PointCloud<PointXYZI>::Ptr pose_keyframe; //position of keyframe (only position)
    
    // Usage for PointPoseInfo
    // position: x, y, z
    // orientation: qw - w, qx - x, qy - y, qz - z
    pcl::PointCloud<PointPoseInfo>::Ptr pose_info_keyframe, pose_info_keyframe_batch; //pose of keyframe (should be denser)

    pcl::PointCloud<PointXYZI>::Ptr pose_each_frame; //position of each frame
    pcl::PointCloud<PointPoseInfo>::Ptr pose_info_each_frame; //pose of each frame

    PointXYZI select_pose; //
    PointType pt_in_local, pt_in_map; // ?? locam point,

    pcl::PointCloud<PointType>::Ptr global_map; // global map to be published
    pcl::PointCloud<PointType>::Ptr global_map_ds;

    vector<pcl::PointCloud<PointType>::Ptr> surf_frames;

    deque<pcl::PointCloud<PointType>::Ptr> recent_surf_keyframes;
    int latest_frame_idx;

    pcl::KdTreeFLANN<PointType>::Ptr kd_tree_surf_local_map;
    pcl::KdTreeFLANN<PointXYZI>::Ptr kd_tree_his_key_poses;

    vector<int> pt_search_idx;
    vector<float> pt_search_sq_dists;

    /* voxelgrid filter */
    pcl::VoxelGrid<PointType> ds_filter_surf;
    pcl::VoxelGrid<PointType> ds_filter_surf_map;
    pcl::VoxelGrid<PointType> ds_filter_his_frames;
    pcl::VoxelGrid<PointType> ds_filter_global_map;

    vector<int> vec_surf_res_cnt; // how many feature correspondance in each frame
    map<int, map<int, int>> gl_vec_surf_res_cnt;
    map<int, map<int, int>> gl_vec_surf_res_cnt_startend;

    // Form of the transformation
    vector<double> abs_pose;
    vector<double> last_pose;

    mutex mutual_exclusion;

    int max_num_iter;

    /* loop closure and back end optimization related stuffs */
    bool loop_closure_on;

    gtsam::NonlinearFactorGraph global_pose_graph;
    gtsam::NonlinearFactorGraph local_pose_graph;
    gtsam::Values global_init_estimate, local_init_estimate;
    gtsam::ISAM2 *isam;
    gtsam::Values global_estimated;

    gtsam::Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    gtsam::noiseModel::Diagonal::shared_ptr prior_noise;
    gtsam::noiseModel::Diagonal::shared_ptr odom_noise;
    gtsam::noiseModel::Diagonal::shared_ptr constraint_noise;

    PointPoseInfo last_GNSS_add_pos;

    // Loop closure detection related
    bool loop_to_close;
    int closest_his_idx;
    int latest_frame_idx_loop;
    bool loop_closed;

    // loosely coupled
    bool GNSSAdded;

    /* back end optimization related stuffs */
    int local_map_width;
    double lc_search_radius;
    int lc_map_width;
    float lc_icp_thres;
    double surfDSRange;

    int slide_window_width; // size of the sliding window
    bool enable_batch_fusion = false;
    int sms_fusion_level = 0;

    double gnssCovThreshold = 200;
    double poseCovThreshold = 1;

    //index of keyframe
    vector<int> keyframe_idx; // accumulate all the new keyframes
    vector<int> keyframe_id_in_frame;

    vector<double> keyframe_time; //

    vector<vector<double>> abs_poses;

    int num_kf_sliding;

    /* imu related */
    vector<sensor_msgs::ImuConstPtr> imu_buf;
    nav_msgs::Odometry::ConstPtr odom_cur;
    vector<nav_msgs::Odometry::ConstPtr> each_odom_buf;
    double time_last_imu;
    double cur_time_imu;
    bool first_imu;
    vector<Preintegration*> pre_integrations;
    Eigen::Vector3d acc_0, gyr_0, g, tmp_acc_0, tmp_gyr_0;

    /* variables to save the states inside the sliding window*/
    vector<Eigen::Vector3d> Ps; // position 
    vector<Eigen::Vector3d> Vs; // velocity 
    vector<Eigen::Matrix3d> Rs; // rotation
    vector<Eigen::Vector3d> Bas; // bias of accelemeters
    vector<Eigen::Vector3d> Bgs; // bias of gyro
    vector<vector<double>> para_speed_bias; // speed and bias?
    vector<vector<double>> rcv_dt;

    //extrinsic imu boady frame to lidar
    Eigen::Quaterniond q_lb;
    Eigen::Vector3d t_lb;

    Eigen::Quaterniond q_bl;
    Eigen::Vector3d t_bl;

    double ql2b_w, ql2b_x, ql2b_y, ql2b_z, tl2b_x, tl2b_y, tl2b_z;

    int idx_imu;
    double gravity;
    
    // GNSS related
    /* lever arm translation */
    Eigen::Matrix4d T_gnss_imu = Eigen::Matrix4d::Identity();

    /* gnss measurements from reference station 
    * the time frame should at least 1 second (<1Hz)
    */
    double station_x_, station_y_, station_z_;

    std::map<int, nlosExclusion::GNSS_Raw_Array> station_gnss_raw_map;
    std::map<int, nlosExclusion::GNSS_Raw_Array> gnss_raw_map;
    GNSS_Tools m_GNSS_Tools;
    ros::Subscriber sub_rtkpos_odometry_float, sub_rtklib_gnss_meas, sub_rtklib_ref_gnss_meas;

    std::vector<std::vector<ObsPtr>> gnss_meas_vec; // vector to save all the epochs of GNSS data (already packed with station and rover GNSS measurements)
    std::mutex m_buf;
    double latest_gnss_time = 0;
    double last_gnss_time = 0;

    Eigen::Vector3d anc_ecef, lever_arm_t;
    Eigen::Matrix3d R_ecef_enu;
    double yaw_enu_local;
    double initial_rpy_[3] = {0};
    Eigen::Quaterniond initial_Quat = Eigen::Quaterniond::Identity();
    double timeshift_IMUtoGNSS;

    double para_anc_ecef[3]; // anchor point in ECEF frame
    double para_yaw_enu_local[1]; // yaw offset from local to enu
    double para_rcv_ddt[EPOCH_SIZE]; // receiver clock bias drift
    double gl_para_rcv_ddt[EPOCH_SIZE]; // receiver clock bias drift

    Eigen::Matrix3d R_enu_local = Eigen::Matrix3d::Identity();
    Eigen::Vector3d ecef_pos, enu_pos, enu_ypr;

    bool GTinLocal, LCinLocal, RTKinLocal; // visualization

    deque<nav_msgs::Odometry> GNSSQueue;

    /* DD measurements related */
    std::map<int, int> epoch_time_idx; // epoch time and correspondence epoch index
    int epoch_idx = 0; // epoch index updated by newly coming signal
    std::vector<std::vector<double>> gt_time_poses; //vector (time, pose)

    Eigen::Vector3d m_pose_enu; // pose in enu from initialization
    std::string result_path, tc_sw_result_path, lc_result_path, batch_result_path;
    std::string result_path_evo, batch_result_path_evo, lc_result_path_evo;

    /* subscribe the ground truth */
    ros::Subscriber span_BP_sub;
    GNSS_Tools gnss_tools_1;
    Eigen::MatrixXd original1;  //referene point in llh 
    bool flag=0; // get the reference point or not? double original_heading = 0;
    nav_msgs::Odometry spanOdom;
    ros::Publisher pub_path_gt;
    nav_msgs::Path path_gt;
    
    //first sliding window optimazition
    bool first_opt;

    // for marginalization
    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    /* be careful, this is the state to be estimated in ceres solver */
    double **tmpQuat;
    double **tmpTrans;
    double **tmpSpeedBias;
    double **tmp_rcv_dt; // receiver clock bias drift

    /* be careful, this is the state to be estimated in ceres solver */
    double **gl_tmpQuat;
    double **gl_tmpTrans;
    double **gl_tmpSpeedBias;
    double **gl_tmp_rcv_dt; // receiver clock bias drift

    bool marg = true;

    vector<int> imu_idx_in_kf;
    double time_last_loop = 0;

    string imu_topic;

    double surf_dist_thres;
    double kd_max_radius;
    bool save_pcd = false;


    double lidar_const = 0;
    int mapping_interval = 1;
    double lc_time_thres = 30.0;
    int start_idx = 0;

    string frame_id = "GLIO";
    string data_set;
    double runtime = 0;

public:
    Estimator(): nh("~") {
        // get result path from current namespace
        if (nh.getParam("result_path", result_path)){
          std::cout<<"result_path para"<<result_path<<std::endl;
        }
        initializeParameters();
        allocateMemory();

        pub_sub_gl_map = nh.advertise<sensor_msgs::PointCloud2>("/sub_global_cloud", 100);

        sub_full_cloud = nh.subscribe<sensor_msgs::PointCloud2>("/full_point_cloud", 100, &Estimator::full_cloudHandler, this);
        sub_surf = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, &Estimator::surfaceLastHandler, this);
        sub_odom = nh.subscribe<nav_msgs::Odometry>("/odom", 10, &Estimator::odomHandler, this);
        sub_each_odom = nh.subscribe<nav_msgs::Odometry>("/each_odom", 10, &Estimator::eachOdomHandler, this);

        sub_imu = nh.subscribe<sensor_msgs::Imu>(imu_topic, 2000, &Estimator::imuHandler, this);

        pub_map = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 2);
        pub_odom = nh.advertise<nav_msgs::Odometry>("/odom_mapped", 2);
        pub_batch_path = nh.advertise<nav_msgs::Path>("/path_batch", 5000);
        pub_poses = nh.advertise<sensor_msgs::PointCloud2>("/trajectory", 2);
        pub_surf = nh.advertise<sensor_msgs::PointCloud2>("/map_surf_less_flat", 2);
        pub_full = nh.advertise<sensor_msgs::PointCloud2>("/raw_scan", 2);

        pub_odom_rtk_ini = nh.advertise<nav_msgs::Odometry>("/odom_rtk_ini", 2);
        pub_path_rtk_ini = nh.advertise<nav_msgs::Path>("rtk_ini_path", 1000);
        pub_gnss_lla = nh.advertise<sensor_msgs::NavSatFix>("gnss_fused_lla", 1000);
        pub_tc_enu_path = nh.advertise<nav_msgs::Path>("tc_enu_path", 1000);
        pub_lc_enu_path = nh.advertise<nav_msgs::Path>("lc_enu_path", 1000);

        /* subscribe rtklib float solution */
        sub_rtkpos_odometry_float = nh.subscribe("/gnss_preprocessor_node/ECEFSolutionRTK", 500, &Estimator::rtklibOdomHandler, this);

        /* subscribe GNSS raw data from rover */
        sub_rtklib_gnss_meas = nh.subscribe("/gnss_preprocessor_node/GNSSPsrCarRov1", 200, &Estimator::rtklib_gnss_meas_callback, this);

        /* subscribe GNSS raw data from reference station */
        sub_rtklib_ref_gnss_meas = nh.subscribe("/gnss_preprocessor_node/GNSSPsrCarStation1", 200, &Estimator::rtklib_ref_gnss_meas_callback, this);
       
        // publish the gt data by reading txt /novatel_data/inspvax inspvax100hz
        span_BP_pub = nh.advertise<novatel_msgs::INSPVAX>("/novatel_data/inspvax", 1000);
        /* subscribe to span-cpt message */
        if (data_set == "HK") {
            span_BP_sub =nh.subscribe("/novatel_data/inspvax", 500, &Estimator::span_bp_callback, this);
        }
        pub_path_gt = nh.advertise<nav_msgs::Path>("path_gt", 1000);
    }

    ~Estimator() {}

    /* allocate the memory for the variables, very very important */
    void allocateMemory() {
        tmpQuat = new double *[slide_window_width]; // orientation
        tmpTrans = new double *[slide_window_width]; // translation
        tmpSpeedBias = new double *[slide_window_width];  // speed, bias_a, bias_g
        tmp_rcv_dt = new double *[slide_window_width];
        for (int i = 0; i < slide_window_width; ++i) {
            tmpQuat[i] = new double[4]; // w, x, y, z
            tmpTrans[i] = new double[3]; // x, y, z
            tmpSpeedBias[i] = new double[9]; // V,BA,BG
            tmp_rcv_dt[i] = new double[3];
        }
        surf_last.reset(new pcl::PointCloud<PointType>());
        surf_local_map.reset(new pcl::PointCloud<PointType>());
        surf_global_map.reset(new pcl::PointCloud<PointType>());
        surf_last_ds.reset(new pcl::PointCloud<PointType>());
        surf_local_map_ds.reset(new pcl::PointCloud<PointType>());
        surf_global_map_ds.reset(new pcl::PointCloud<PointType>());
        full_cloud.reset(new pcl::PointCloud<PointType>());


        /* vectors related to the factors construction */
        for(int i = 0; i < slide_window_width; ++i) {
            pcl::PointCloud<PointType>::Ptr tmpSurfCurrent;
            tmpSurfCurrent.reset(new pcl::PointCloud<PointType>());
            vec_surf_cur_pts.push_back(tmpSurfCurrent);

            vector<double> tmpD;
            vec_surf_scores.push_back(tmpD);

            pcl::PointCloud<PointType>::Ptr tmpSurfNorm;
            tmpSurfNorm.reset(new pcl::PointCloud<PointType>());
            vec_surf_normal.push_back(tmpSurfNorm);

            vec_surf_res_cnt.push_back(0);
        }

        pose_keyframe.reset(new pcl::PointCloud<PointXYZI>());
        pose_info_keyframe.reset(new pcl::PointCloud<PointPoseInfo>());
        pose_info_keyframe_batch.reset(new pcl::PointCloud<PointPoseInfo>());

        pose_each_frame.reset(new pcl::PointCloud<PointXYZI>());
        pose_info_each_frame.reset(new pcl::PointCloud<PointPoseInfo>());

        global_map.reset(new pcl::PointCloud<PointType>());
        global_map_ds.reset(new pcl::PointCloud<PointType>());

        latest_key_frames.reset(new pcl::PointCloud<PointType>());
        latest_key_frames_ds.reset(new pcl::PointCloud<PointType>());
        his_key_frames.reset(new pcl::PointCloud<PointType>());
        his_key_frames_ds.reset(new pcl::PointCloud<PointType>());

        kd_tree_surf_local_map.reset(new pcl::KdTreeFLANN<PointType>());
        kd_tree_his_key_poses.reset(new pcl::KdTreeFLANN<PointXYZI>());
    }

    /* get parameters from the .yaml file */
    void initializeParameters() {
        gtsam::ISAM2Params isamPara;
        isamPara.relinearizeThreshold = 0.1;
        isamPara.relinearizeSkip = 1;
        isam = new gtsam::ISAM2(isamPara);

        // initialize the GNSS related states
        anc_ecef.setZero();
        lever_arm_t.setZero();
        R_ecef_enu.setIdentity();
        poseCovariance.setZero();
        para_yaw_enu_local[0] = 0;
        yaw_enu_local = 0;
        m_pose_enu = Eigen::Vector3d(0, 0, 0);
        last_GNSS_add_pos.x = 0;
        last_GNSS_add_pos.y = 0;
        last_GNSS_add_pos.z = 0;

        // Load parameters from yaml
        if (!getParameter("/initialization/anc_ecef_x", anc_ecef(0))) {
            ROS_WARN("data_set not set, use default value: 0");
            anc_ecef(0) = 0;
        }
        else {
            rtk_infos.data.push_back(anc_ecef(0));
        }

        if (!getParameter("/initialization/anc_ecef_y", anc_ecef(1))) {
            ROS_WARN("data_set not set, use default value: 0");
            anc_ecef(1) = 0;
        }
        else {
            rtk_infos.data.push_back(anc_ecef(1));
        }

        if (!getParameter("/initialization/anc_ecef_z", anc_ecef(2))) {
            ROS_WARN("data_set not set, use default value: 0");
            anc_ecef(2) = 0;
        }
        else {
            rtk_infos.data.push_back(anc_ecef(2));
        }

        if (!getParameter("/initialization/yaw_enu_local", yaw_enu_local)) {
            ROS_WARN("data_set not set, use default value: 0");
            yaw_enu_local = 0;
        }
        yaw_enu_local = yaw_enu_local * (3.14159/180.0);

        if (!getParameter("/initialization/Euler_r", initial_rpy_[0])) {
            ROS_WARN("data_set not set, use default value: 0");
        }
        if (!getParameter("/initialization/Euler_p", initial_rpy_[1])) {
            ROS_WARN("data_set not set, use default value: 0");
        }
        if (!getParameter("/initialization/Euler_y", initial_rpy_[2])) {
            ROS_WARN("data_set not set, use default value: 0");
        }

        if (!getParameter("/initialization/lever_arm_x", lever_arm_t(0))) {
            ROS_WARN("data_set not set, use default value: 0");
            lever_arm_t(0) = 0;
        }
        if (!getParameter("/initialization/lever_arm_y", lever_arm_t(1))) {
            ROS_WARN("data_set not set, use default value: 0");
            lever_arm_t(1) = 0;
        }
        if (!getParameter("/initialization/lever_arm_z", lever_arm_t(2))) {
            ROS_WARN("data_set not set, use default value: 0");
            lever_arm_t(2) = 0;
        }

        if (!getParameter("/initialization/timeshift", timeshift_IMUtoGNSS)) {
            ROS_WARN("data_set not set, use default value: 0");
            timeshift_IMUtoGNSS = 0;
        }

        if (!getParameter("/initialization/station_x_", station_x_)) {
            ROS_WARN("station_x_ not set, use default value: 0");
            station_x_ = 0;
        }
        else {
            rtk_infos.data.push_back(station_x_);
        }

        if (!getParameter("/initialization/station_y_", station_y_)) {
            ROS_WARN("station_y_ not set, use default value: 0");
            station_y_ = 0;
        }
        else {
            rtk_infos.data.push_back(station_y_);
        }

        if (!getParameter("/initialization/station_z_", station_z_)) {
            ROS_WARN("station_z_ not set, use default value: 0");
            station_z_ = 0;
        }
        else {
            rtk_infos.data.push_back(station_z_);
        }

        R_ecef_enu = ecef2rotation(anc_ecef);
        para_yaw_enu_local[0] = yaw_enu_local;

        // while(para_yaw_enu_local[0] > M_PI)   para_yaw_enu_local[0] -= 2.0*M_PI;
        // while(para_yaw_enu_local[0] < -M_PI)  para_yaw_enu_local[0] += 2.0*M_PI;

        para_anc_ecef[0] = anc_ecef(0);
        para_anc_ecef[1] = anc_ecef(1);
        para_anc_ecef[2] = anc_ecef(2);

        for (int i=0; i<EPOCH_SIZE; i++) {
            para_rcv_ddt[i] = 0;
            gl_para_rcv_ddt[i] = 0;
        }

        // Load parameters from yaml
        if (!getParameter("/common/data_set", data_set)) {
            ROS_WARN("data_set not set, use default value: utbm");
            data_set = "utbm";
        }

        if (!getParameter("/Estimator/enable_batch_fusion", enable_batch_fusion)) {
            ROS_WARN("enable_batch_fusion not set, use default value: false");
            enable_batch_fusion = false;
        }

        if (!getParameter("/Estimator/sms_fusion_level", sms_fusion_level)) {
            ROS_WARN("sms_fusion_level not set, use default value: 0");
            sms_fusion_level = 0;
        }

        if (!getParameter("/Estimator/surf_dist_thres", surf_dist_thres)) {
            ROS_WARN("surf_dist_thres not set, use default value: 0.1");
            surf_dist_thres = 0.1;
        }

        if (!getParameter("/Estimator/kd_max_radius", kd_max_radius)) {
            ROS_WARN("kd_max_radius not set, use default value: 1.0");
            kd_max_radius = 1.0;
        }

        if (!getParameter("/Estimator/save_pcd", save_pcd)) {
            ROS_WARN("save_pcd not set, use default value: false");
            save_pcd = false;
        }

        if (!getParameter("/Estimator/mapping_interval", mapping_interval)) {
            ROS_WARN("mapping_interval not set, use default value: 1");
            mapping_interval = 1;
        }

        if (!getParameter("/Estimator/lc_time_thres", lc_time_thres)) {
            ROS_WARN("lc_time_thres not set, use default value: 30.0");
            lc_time_thres = 30.0;
        }

        if (!getParameter("/Estimator/lidar_const", lidar_const)) {
            ROS_WARN("lidar_const not set, use default value: 1.0");
            lidar_const = 1.0;
        }

        if (!getParameter("/IMU/imu_topic", imu_topic)) {
            ROS_WARN("imu_topic not set, use default value: /imu/data");
            imu_topic = "/imu/data";
        }

        if (!getParameter("/Estimator/max_num_iter", max_num_iter)) {
            ROS_WARN("maximal iteration number of mapping optimization not set, use default value: 50");
            max_num_iter = 50;
        }

        if (!getParameter("/Estimator/loop_closure_on", loop_closure_on)) {
            ROS_WARN("loop closure detection set to false");
            loop_closure_on = false;
        }

        if (!getParameter("/Estimator/local_map_width", local_map_width)) {
            ROS_WARN("local_map_width not set, use default value: 5");
            local_map_width = 5;
        }

        if (!getParameter("/Estimator/lc_search_radius", lc_search_radius)) {
            ROS_WARN("lc_search_radius not set, use default value: 7.0");
            lc_search_radius = 7.0;
        }

        if (!getParameter("/Estimator/lc_map_width", lc_map_width)) {
            ROS_WARN("lc_map_width not set, use default value: 25");
            lc_map_width = 25;
        }

        if (!getParameter("/Estimator/lc_icp_thres", lc_icp_thres)) {
            ROS_WARN("lc_icp_thres not set, use default value: 0.3");
            lc_icp_thres = 0.3;
        }

        if (!getParameter("/Estimator/surfDSRange", surfDSRange)) {
            ROS_WARN("surfDSRange not set, use default value: 0.4");
            surfDSRange = 0.4;
        }

        if (!getParameter("/Estimator/slide_window_width", slide_window_width)) {
            ROS_WARN("slide_window_width not set, use default value: 4");
            slide_window_width = 4;
        }

        if (!getParameter("/Estimator/gnssCovThreshold", gnssCovThreshold)) {
            ROS_WARN("gnssCovThreshold not set, use default value: 200");
            gnssCovThreshold = 200;
        }

        if (!getParameter("/Estimator/poseCovThreshold", poseCovThreshold)) {
            ROS_WARN("poseCovThreshold not set, use default value: 1");
            poseCovThreshold = 1;
        }

        //load feature selection parameter
        if (!getParameter("/feature_selection/feature_res_num", feature_res_num)) {
            ROS_WARN("selected feature number is not set");
            feature_res_num = 60;
        }

        if (!getParameter("/feature_selection/rand_set_num", rand_set_num)) {
            ROS_WARN("point number of random set is not set");
            rand_set_num = 300;
        }

        if (!getParameter("/feature_selection/batch_feature_res_num", batch_feature_res_num)) {
            ROS_WARN("selected feature number is not set");
            batch_feature_res_num = 60;
        }

        if (!getParameter("/feature_selection/batch_rand_set_num", batch_rand_set_num)) {
            ROS_WARN("point number of random set is not set");
            batch_rand_set_num = 300;
        }

        if (!getParameter("/feature_selection/random_select", random_select)) {
            ROS_WARN("random select mode set to false");
            random_select = false;
        }

        //extrinsic parameters
        if (!getParameter("/Estimator/ql2b_w", ql2b_w))  {
            ROS_WARN("ql2b_w not set, use default value: 1");
            ql2b_w = 1;
        }

        if (!getParameter("/Estimator/ql2b_x", ql2b_x)) {
            ROS_WARN("ql2b_x not set, use default value: 0");
            ql2b_x = 0;
        }

        if (!getParameter("/Estimator/ql2b_y", ql2b_y)) {
            ROS_WARN("ql2b_y not set, use default value: 0");
            ql2b_y = 0;
        }

        if (!getParameter("/Estimator/ql2b_z", ql2b_z)) {
            ROS_WARN("ql2b_z not set, use default value: 0");
            ql2b_z = 0;
        }

        if (!getParameter("/Estimator/tl2b_x", tl2b_x)) {
            ROS_WARN("tl2b_x not set, use default value: 0");
            tl2b_x = 0;
        }

        if (!getParameter("/Estimator/tl2b_y", tl2b_y)) {
            ROS_WARN("tl2b_y not set, use default value: 0");
            tl2b_y = 0;
        }

        if (!getParameter("/Estimator/tl2b_z", tl2b_z)) {
            ROS_WARN("tl2b_z not set, use default value: 0");
            tl2b_z = 0;
        }

        if (!getParameter("/Estimator/search_range", search_range)) {
            ROS_WARN("search_range not set, use default value: 0");
            search_range = 0;
        }

        if (!getParameter("/visualization/GTinLocal", GTinLocal)) {
            ROS_WARN("GTinLocal not set, use default value: false");
            GTinLocal = false;
        }

        if (!getParameter("/visualization/LCinLocal", LCinLocal)) {
            ROS_WARN("LCinLocal not set, use default value: false");
            LCinLocal = false;
        }

        if (!getParameter("/visualization/RTKinLocal", RTKinLocal)) {
            ROS_WARN("RTKinLocal not set, use default value: false");
            RTKinLocal = false;
        }

        tc_sw_result_path = result_path + "tc_sw_result.csv";
        lc_result_path = result_path + "lc_result.csv";
//        result_path_evo = result_path + "enu_q_evo.csv";
//        batch_result_path_evo = result_path + "batch_enu_q_evo.csv";
//        lc_result_path_evo = result_path + "lc_enu_q_evo.csv";

        // clear output file
        std::ofstream tc_sw_output(tc_sw_result_path, std::ios::out);
        tc_sw_output.close();
        std::ofstream lc_output(lc_result_path, std::ios::out);
        lc_output.close();
//        std::ofstream res_evo_output(result_path_evo, std::ios::out);
//        res_evo_output.close();
//        std::ofstream res_lc_evo_output(lc_result_path_evo, std::ios::out);
//        res_lc_evo_output.close();

        last_marginalization_info = nullptr;

        idx_imu = 0;
        first_opt = false;
        cur_time_imu = -1;

        Rs.push_back(Eigen::Matrix3d::Identity());
        Ps.push_back(Eigen::Vector3d(0, 0, 0)); // Eigen::Vector3d(-0.9, -0.2, 0.0)
        Vs.push_back(Eigen::Vector3d(0, 0, 0));

        Bas.push_back(Eigen::Vector3d::Zero());
        Bgs.push_back(Eigen::Vector3d(0, 0, 0));
        vector<double> tmpSpeedBias;
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);

        para_speed_bias.push_back(tmpSpeedBias);

        num_kf_sliding = 0;
        time_last_imu = 0;
        first_imu = false;

        nh.param<double>("/IMU/gravity", gravity, 9.805);
        g = Eigen::Vector3d(0, 0, gravity); //

        time_new_odom = 0;

        abs_pose.push_back(1);
        last_pose.push_back(1);

        latest_frame_idx = 0;

        vector<double> tmpOdom;
        tmpOdom.push_back(1);

        for (int i = 1; i < 7; ++i) {
            abs_pose.push_back(0);
            last_pose.push_back(0);
            tmpOdom.push_back(0);
        }
        abs_poses.push_back(tmpOdom);

        abs_pose = tmpOdom;

        /* when we integrate the GNSS, less LiDAR features are needed */
        ds_filter_surf.setLeafSize(surfDSRange, surfDSRange, surfDSRange);

        ds_filter_surf_map.setLeafSize(0.4, 0.4, 0.4);
        ds_filter_his_frames.setLeafSize(0.4, 0.4, 0.4);
        ds_filter_global_map.setLeafSize(0.2, 0.2, 0.2);

        odom_mapping.header.frame_id = frame_id;
        odom_init_kf.header.frame_id = frame_id;

        /* settings related to GTSAM iSAM */
        gtsam::Vector vector6p(6);
        gtsam::Vector vector6o(6);
        vector6p << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8; //1e-6, 1e-6, 1e-6
        vector6o << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
        prior_noise = gtsam::noiseModel::Diagonal::Variances(vector6p);
        odom_noise = gtsam::noiseModel::Diagonal::Variances(vector6o);

        loop_to_close = false;
        loop_closed = false;

        /* extrinsic parameters */
        q_lb = Eigen::Quaterniond(ql2b_w, ql2b_x, ql2b_y, ql2b_z);
        t_lb = Eigen::Vector3d(tl2b_x, tl2b_y, tl2b_z);

        q_bl = q_lb.inverse();
        t_bl = - (q_bl * t_lb);

        /* init lever arm translation */
        T_gnss_imu.block<3, 1>(0, 3) = lever_arm_t;
    }

    /**
    * @brief span_cpt callback
    * @param span_cpt bestpos msg
    * @return void
    @ 
    */
    void span_bp_callback(const novatel_msgs::INSPVAXConstPtr& fix_msg)
    {
        if(GTinLocal)
        {
            gtime_t gt_gps_time = gpst2time(fix_msg->header.gps_week, fix_msg->header.gps_week_seconds/1000.0);
            double latest_gnss_time = time2sec(gt_gps_time);

            Eigen::MatrixXd lolaal;
            lolaal.resize(3,1);
            lolaal(0) = fix_msg->longitude;
            lolaal(1) = fix_msg->latitude;
            lolaal(2) = fix_msg->altitude;

            Eigen::MatrixXd ecef;
            ecef.resize(3,1);
            ecef=gnss_tools_1.llh2ecef(lolaal);

            original1.resize(3,1);
            if(flag==0) // the initial value
            {
                flag=1;
                original1(0)=lolaal(0);
                original1(1)=lolaal(1);
                original1(2)=lolaal(2);
            }

            /* get the ENU solution from SPAN-CPT */
            Eigen::MatrixXd enu1;
            enu1.resize(3,1);
            enu1= gnss_tools_1.ecef2enu(original1, ecef);

            double roll = fix_msg->roll;
            double pitch = fix_msg->pitch;
            double yaw = -fix_msg->azimuth;
            tf2::Quaternion gt_q;
            gt_q.setRPY(pitch * 3.1415926/180, roll * 3.1415926/180, yaw * 3.1415926/180);
            gt_q.normalize();

            /* publish the odometry from span-cpt */
            spanOdom.header.frame_id = "GLIO";
            spanOdom.child_frame_id = "GLIO";
            spanOdom.pose.pose.position.x = enu1(0);
            spanOdom.pose.pose.position.y = enu1(1);
            spanOdom.pose.pose.position.z = enu1(2);
            spanOdom.pose.pose.orientation.w = gt_q.w();
            spanOdom.pose.pose.orientation.x = gt_q.x();
            spanOdom.pose.pose.orientation.y = gt_q.y();
            spanOdom.pose.pose.orientation.z = gt_q.z();

            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header = spanOdom.header;
            pose_stamped.header.frame_id = "GLIO";
            pose_stamped.pose = spanOdom.pose.pose;
            path_gt.header = spanOdom.header;
            path_gt.header.frame_id = "GLIO";
            path_gt.poses.push_back(pose_stamped);
            pub_path_gt.publish(path_gt);
        }
    }

    /* subscribe the gnss data from reference station */
    void rtklib_ref_gnss_meas_callback(const nlosExclusion::GNSS_Raw_ArrayConstPtr &gnss_msg)
    {
        m_buf.lock();
        /* if the GNSS data from the reference station is correct, the estimated
        position from single point positioning should be close the the truth position within 10 meters */
        Eigen::MatrixXd eWLSSolutionECEF = m_GNSS_Tools.WeightedLeastSquare( //WeightedLeastSquareGPSBDS
                m_GNSS_Tools.getAllPositions(*gnss_msg),
                m_GNSS_Tools.getAllMeasurements(*gnss_msg),
                *gnss_msg, "WLS");
        Eigen::Matrix<double ,3,1> WLSENU;
        WLSENU << 0, 0, 0;
        /* reference point for ENU calculation */
        Eigen::Matrix<double, 3,1> ENU_ref;
        ENU_ref << 0, 0, 0;
        Eigen::Vector3d base_pose  = Eigen::Vector3d(station_x_, station_y_, station_z_);
        ENU_ref = m_GNSS_Tools.ecef2llh(base_pose);
        WLSENU = m_GNSS_Tools.ecef2enu(ENU_ref, eWLSSolutionECEF);

        double disNorm = sqrt(pow(WLSENU(0), 2) + pow(WLSENU(1), 2) + pow(WLSENU(2), 2));

        if(true) //disNorm< 30
        {
            gtime_t gps_time_ = gpst2time(gnss_msg->GNSS_Raws[0].GNSS_week, gnss_msg->GNSS_Raws[0].GNSS_time);
            double gps_time = time2sec(gps_time_);
            station_gnss_raw_map[round((gps_time - 1600000000 - timeshift_IMUtoGNSS)*10)] = *gnss_msg;
        }

        m_buf.unlock();
    }

    /* transform the gnss raw data to map format */
    bool gnssRawArray2map(nlosExclusion::GNSS_Raw_Array gnss_data, std::map<int, nlosExclusion::GNSS_Raw> &epochGnssMap)
    {
        for(int i = 0; i < gnss_data.GNSS_Raws.size(); i++)
        {
            int prn = int(gnss_data.GNSS_Raws[i].prn_satellites_index);
            epochGnssMap[prn] = gnss_data.GNSS_Raws[i];
        }
        return true;
    }

    /* master satellite involved in both station and rcv gnss
    the master satellite and the target prn should from same satellite systems (GPS or BeiDou)
    */
    bool findMasterPrn(std::map<int, nlosExclusion::GNSS_Raw> epochRcvGNSSMap, std::map<int, nlosExclusion::GNSS_Raw> epochStGNSSMap, int &masterPrn, int targetPrn)
    {
        string sys_tg;
        if (m_GNSS_Tools.PRNisGPS(targetPrn)) sys_tg = "GPS";
        else if (m_GNSS_Tools.PRNisBeidou(targetPrn)) sys_tg = "BDS";
        else if (m_GNSS_Tools.PRNisGLONASS(targetPrn)) sys_tg = "GLO";
        else if (m_GNSS_Tools.PRNisGAL(targetPrn)) sys_tg = "GAL";
        else
            return false;
        double maxEle = -1;

        for (auto ti : epochRcvGNSSMap)
        {
            int findFlag = epochStGNSSMap.count(ti.first);

            string sys_id;
            if (m_GNSS_Tools.PRNisGPS(ti.first)) sys_id = "GPS";
            else if (m_GNSS_Tools.PRNisBeidou(ti.first)) sys_id = "BDS";
            else if (m_GNSS_Tools.PRNisGLONASS(ti.first)) sys_id = "GLO";
            else if (m_GNSS_Tools.PRNisGAL(ti.first)) sys_id = "GAL";
            if (sys_id == sys_tg && findFlag && (ti.second.elevation) > maxEle ) //&& (ti.second.carrier_phase>100)
            {
                maxEle = ti.second.elevation;
                masterPrn = ti.first;
            }
        } 
        return true;
    }

    /* convert gnss raw to single Obs struct */
    bool gnssraw2SO(nlosExclusion::GNSS_Raw gnss_data, singleObs &SO)
    {
        SO.prn_satellites_index = gnss_data.prn_satellites_index;
        SO.pseudorange          = gnss_data.pseudorange;
        SO.raw_pseudorange      = gnss_data.raw_pseudorange;
        SO.carrier_phase        = gnss_data.carrier_phase;
        SO.doppler              = gnss_data.doppler;
        SO.lamda                = gnss_data.lamda;
        SO.snr                  = gnss_data.snr;
        SO.elevation            = gnss_data.elevation;
        SO.azimuth              = gnss_data.azimuth;
        SO.err_tropo            = gnss_data.err_tropo;
        SO.err_iono             = gnss_data.err_iono;
        SO.sat_clk_err          = gnss_data.sat_clk_err;
        SO.sat_pos_x            = gnss_data.sat_pos_x;
        SO.sat_pos_y            = gnss_data.sat_pos_y;
        SO.sat_pos_z            = gnss_data.sat_pos_z;
        SO.ttx                  = gnss_data.ttx;
        SO.vel_x                = gnss_data.vel_x;
        SO.vel_y                = gnss_data.vel_y;
        SO.vel_z                = gnss_data.vel_z;
        SO.dt                   = gnss_data.dt;
        SO.ddt                  = gnss_data.ddt;
        SO.tgd                  = gnss_data.tgd;
        SO.visable              = gnss_data.visable;
        SO.sat_system           = gnss_data.sat_system;
        SO.visable3DMA          = gnss_data.visable3DMA;
        SO.prE3dMA              = gnss_data.prE3dMA;
        SO.slip                 = gnss_data.slip;
    }

    /* calculate the variance (sigma square) */
    bool calVar(ObsPtr &obs)
    {
        double var = 0;

        /* variance for r_i_sat */
        obs->r_i_sat.var = eleSRNVarCal(obs->r_i_sat.elevation, obs->r_i_sat.snr);

        /* variance for r_m_sat */
        obs->r_m_sat.var = eleSRNVarCal(obs->r_m_sat.elevation, obs->r_m_sat.snr);

        /* variance for u_m_sat */
        obs->u_m_sat.var = eleSRNVarCal(obs->u_m_sat.elevation, obs->u_m_sat.snr);

        /* variance for u_i_sat */
        obs->var = eleSRNVarCal(obs->Uelevation, obs->Usnr);

        return true;
    }

    /**
     * @brief variance estimation (sigma square)
     * @param nlosExclusion::GNSS_Raw_Array GNSS_data
     * @return weight_matrix
     @ 
    */
    double eleSRNVarCal(double ele, double snr)
    {
        int model = 0; // 0: ele&&SNR model. 1: RTKLIB ele model. 2: DLR Ele model
        Eigen::Matrix<double,4,1> parameters;
        parameters<<50.0, 30.0, 30.0, 10.0; // loosely coupled 
        // parameters<<50.0, 30.0, 20.0, 30.0; // loosely coupled 
        double snr_1 = parameters(0); // T = 50
        double snr_A = parameters(1); // A = 30
        double snr_a = parameters(2);// a = 30
        double snr_0 = parameters(3); // F = 10

        double snr_R = snr;
        // snr_R = 40;
        double elR = ele;
        double q_R_1 = 1 / (pow(( sin(elR * 3.1415926/180.0 )),2));
        double q_R_2 = pow(10,(-(snr_R - snr_1) / snr_a));
        double q_R_3 = (((snr_A / (pow(10,(-(snr_0 - snr_1) / snr_a))) - 1) / (snr_0 - snr_1)) * (snr_R - snr_1) + 1);
        double q_R = q_R_1* (q_R_2 * q_R_3);
        
        if(model == 0)
        {
            double weghting =(1.0/float(q_R)); // uncertainty: cofactor_[i] larger, larger uncertainty
            return (1.0/weghting);
        }
        
    }

    /* subscribe the odometry from RTKLIB in ECEF*/
    void rtklibOdomHandler(const nav_msgs::Odometry::ConstPtr& odomIn) {
        if(RTKinLocal)
        {
            Eigen::Matrix<double, 3,1> ENU_ref;
            ENU_ref << 0, 0, 0;
            ENU_ref = m_GNSS_Tools.ecef2llh(anc_ecef);
            Eigen::Matrix<double, 3, 1> ENU;
            Eigen::Matrix<double, 3, 1> ECEF;
            ECEF<<odomIn->pose.pose.position.x, odomIn->pose.pose.position.y, odomIn->pose.pose.position.z;
            ENU = m_GNSS_Tools.ecef2enu(ENU_ref, ECEF);
            if (fabs(ENU(2) > 300)) return;

            if (fabs(fabs(ENU(0)) - fabs(m_pose_enu(0))) > 300 ||
                fabs(fabs(ENU(1)) - fabs(m_pose_enu(1))) > 300 ||
                fabs(fabs(ENU(2)) - fabs(m_pose_enu(2))) > 300) return;

            nav_msgs::Odometry gnssOdom;
            gnssOdom.header.stamp = odomIn->header.stamp;
            gnssOdom.pose.pose.position.x = ENU(0);
            gnssOdom.pose.pose.position.y = ENU(1);
            gnssOdom.pose.pose.position.z = ENU(2);
            gnssOdom.pose.covariance[0] = odomIn->pose.covariance[0];
            gnssOdom.pose.covariance[1] = odomIn->pose.covariance[1];
            gnssOdom.pose.covariance[2] = odomIn->pose.covariance[2];

            GNSSQueue.push_back(gnssOdom);
            m_pose_enu = ENU;

            /* publish the pose from the RTK initialization */
            nav_msgs::Odometry odom_mapping_rtk_ini;
            odom_mapping_rtk_ini = odom_mapping;
            odom_mapping_rtk_ini.pose.pose.position.x = m_pose_enu(0);
            odom_mapping_rtk_ini.pose.pose.position.y = m_pose_enu(1);
            odom_mapping_rtk_ini.pose.pose.position.z = m_pose_enu(2);
            pub_odom_rtk_ini.publish(odom_mapping_rtk_ini);
            rtk_ini_enu_path.header = odom_mapping_rtk_ini.header;
            geometry_msgs::PoseStamped enu_pose_msg;
            enu_pose_msg.header = rtk_ini_enu_path.header;
            enu_pose_msg.pose.position.x = m_pose_enu(0);
            enu_pose_msg.pose.position.y = m_pose_enu(1);
            enu_pose_msg.pose.position.z = m_pose_enu(2);

            rtk_ini_enu_path.poses.push_back(enu_pose_msg);
            pub_path_rtk_ini.publish(rtk_ini_enu_path);
        }
    }

    /* subscribe the gnss range/doppler measurements from rover */
    void rtklib_gnss_meas_callback(const nlosExclusion::GNSS_Raw_ArrayConstPtr &meas_msg)
    {
        /* measurements from station should be available */
        if(station_gnss_raw_map.size() < 1)
        {
            ROS_WARN("waiting for station gnss message!");
            return;
        }
        nlosExclusion::GNSS_Raw_Array meas_msg_valid;
        /* calculate the integer GNSS second to find station GNSS measurements */
        gtime_t gps_time_ = gpst2time(meas_msg->GNSS_Raws[0].GNSS_week, meas_msg->GNSS_Raws[0].GNSS_time);
        double gps_time = time2sec(gps_time_);
        int integerGNSSTime = round((gps_time - 1600000000 - timeshift_IMUtoGNSS)*10);

        std::vector<ObsPtr> gnss_meas;
        int length = meas_msg->GNSS_Raws.size();
        if(length == 0)
        {
            return;
        }  

        double curGNSSSec = meas_msg->GNSS_Raws[0].GNSS_time;

        nlosExclusion::GNSS_Raw_Array closest_gnss_data; // gnss from receiver
        closest_gnss_data = *meas_msg;
        nlosExclusion::GNSS_Raw_Array st_gnss_data; // gnss from station

        /* try to find the station GNSS measurements */
        int integerGNSSTimeTmp = integerGNSSTime;
        std::map<int, nlosExclusion::GNSS_Raw_Array>::iterator iter_pr;
        iter_pr = station_gnss_raw_map.begin();
        int diff = 0, max_diff = fabs(iter_pr->first - integerGNSSTime);
        int ref_idx = iter_pr->first;
        for (int i=0; i<station_gnss_raw_map.size(); i++,iter_pr++) {
            diff = fabs(iter_pr->first - integerGNSSTime);
            if (diff < max_diff) {
                ref_idx = iter_pr->first;
                max_diff = diff;
            }
        }
        if (max_diff < 300) {
            st_gnss_data = station_gnss_raw_map.at(ref_idx);
        }

        /* station GNSS data to map */
        std::map<int, nlosExclusion::GNSS_Raw> epochStGNSSMap; // array to map
        gnssRawArray2map(st_gnss_data, epochStGNSSMap);

        /* rover GNSS data to map (RCV denotes receiver) */
        std::map<int, nlosExclusion::GNSS_Raw> epochRcvGNSSMap; // array to map
        gnssRawArray2map(closest_gnss_data, epochRcvGNSSMap);

        /* find the DD measurements related observation
         * please refer to GraphGNSSLib
         * Wen Weisong., Hsu, Li-Ta.* Towards Robust GNSS Positioning and Real-Time Kinematic Using Factor Graph Optimization, ICRA 2021, Xi'an, China.
        */
        int sv_cnt = st_gnss_data.GNSS_Raws.size();
        /*u_master_sv: user to master 
        *u_iSV: user to ith satellite
        *r_master_sv: reference to master satellite
        */

        int num_DD_psr_sat = 0;
        int num_DD_car_sat = 0;

        /* index the rcv gnss */
        for (size_t i = 0; i < length; i++)
        {
            ObsPtr obs(new Obs());
            
            /* original custimized msg: single satellite */
            nlosExclusion::GNSS_Raw data = meas_msg->GNSS_Raws[i];
            int satPrn = data.prn_satellites_index;
            if (satPrn > 127) continue;
            if(!epochStGNSSMap.count(satPrn)) {
                continue;
            }
            meas_msg_valid.GNSS_Raws.push_back(data);
            // if(data.pseudorange<100) continue;

            /*only do this for double-differencing related stuffs */
            if (true)
            {
                /* during the DD observation, we only use the satellite with high elevation angle */
                if(data.elevation<15)
                {
                    continue;
                }

                /* find the master satellite */
                int masterSatPrn = -1;
                bool result = findMasterPrn(epochRcvGNSSMap, epochStGNSSMap, masterSatPrn, satPrn);
                if(masterSatPrn<0)
                {
                    return;
                }

                nlosExclusion::GNSS_Raw u_master_sv, u_iSV; // from user side
                nlosExclusion::GNSS_Raw r_master_sv, r_iSV; // from reference station side

                u_master_sv = epochRcvGNSSMap.at(masterSatPrn); // rcv master gnss
                r_master_sv = epochStGNSSMap.at(masterSatPrn); // sta master gnss

                singleObs SOu_master_sv, SOr_master_sv, SO_r_iSV;
                gnssraw2SO(u_master_sv, SOu_master_sv);
                gnssraw2SO(r_master_sv, SOr_master_sv);

                /* prepare the DD measurements */
                if(epochStGNSSMap.count(satPrn))
                {
                    r_iSV = epochStGNSSMap.at(satPrn);
                    gnssraw2SO(r_iSV, SO_r_iSV);
                    obs->r_i_sat = SO_r_iSV; // station to ith sat
                    obs->u_m_sat = SOu_master_sv; // rcv to master sat
                    obs->r_m_sat = SOr_master_sv; // station to master sat

                    /* a DD psr measurement */
                    if(satPrn!=masterSatPrn)
                    {
                        num_DD_psr_sat ++;
                        obs->DD_psr = true;

                        /* a DD carrier measurement */
                        if(data.carrier_phase> 100 && SO_r_iSV.carrier_phase > 100 && SOu_master_sv.carrier_phase > 100 && SOr_master_sv.carrier_phase > 100)
                        {
                            num_DD_car_sat++;
                            obs->DD_car = true;
                        }
                    }

                }
                else
                {
                    obs->DD_psr = false;
                    continue;
                }
            }

            /* receiver to ith satellite */
            obs->time       = gpst2time(data.prE3dMA, data.GNSS_time);
            obs->sat        = data.prn_satellites_index;
            obs->freqs.push_back(data.lamda);
            obs->CN0.push_back(data.snr);
            obs->LLI.push_back(0);
            obs->code.push_back(1);
            obs->psr.push_back(data.pseudorange);
            obs->psr_std.push_back(1.0);
            obs->cp.push_back(data.carrier_phase);
            obs->cp_std.push_back(1.0);
            obs->dopp.push_back(data.doppler);
            obs->dopp_std.push_back(1.0);
            obs->status.push_back(*(std::to_string(1).c_str()));
            //set carr phase cycle slip flag
            if (obs->DD_car) {
                uint8_t ds = *(std::to_string(data.slip).c_str());
                obs->status.push_back(ds);
            }

            /* measurements */
            obs->Uvisable = 1;
            obs->Usnr = data.snr;
            obs->Upseudorange = data.pseudorange;
            obs->Uraw_pseudorange = data.raw_pseudorange;
            obs->Ucarrier_phase = data.carrier_phase;
            obs->Ulamda = data.lamda;
            obs->Uprn_satellites_index = data.prn_satellites_index;

            obs->Uelevation = data.elevation;
            obs->Uazimuth = data.azimuth;
            obs->Uerr_iono = data.err_iono;
            obs->Uerr_tropo = data.err_tropo;
            obs->Usat_clk_err = data.sat_clk_err;
            obs->Usat_pos_x = data.sat_pos_x;
            obs->Usat_pos_y = data.sat_pos_y;
            obs->Usat_pos_z = data.sat_pos_z;
            obs->Uttx = data.ttx;
            obs->Uvel_x = data.vel_x;
            obs->Uvel_y = data.vel_y;
            obs->Uvel_z = data.vel_z;
            obs->Udt = data.dt;
            obs->Uddt = data.ddt;
            obs->Utgd = data.tgd;

            calVar(obs); // calculate covariance for four satellite measurements
            
            gnss_meas.push_back(obs);
        }

        if (gnss_meas.size() == 0) return;
        /* form the received gnss sginal vectors */
        gnss_raw_map[integerGNSSTimeTmp] = meas_msg_valid;

        latest_gnss_time = time2sec(gnss_meas[0]->time);

        m_buf.lock();
        /* make sure the order of GNSS data is correct */
        if(latest_gnss_time > last_gnss_time)
        {
            gnss_meas_vec.push_back(gnss_meas);
        }
        else
        {
             std::cout<<"GNSS measurements in disorder:   " << latest_gnss_time - last_gnss_time<<"\n";
        }

        last_gnss_time = latest_gnss_time;
        
        m_buf.unlock();
    }

    /* subscribe the full clouds from the lidar odometry */
    void full_cloudHandler(const sensor_msgs::PointCloud2ConstPtr& pointCloudIn) {
        full_cloud->clear();
        pcl::fromROSMsg(*pointCloudIn, *full_cloud);
        pcl::PointCloud<PointType>::Ptr full(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*full_cloud, *full);
        new_full_cloud = true;
    }

    /* subscribe the surface clouds from the lidar odometry */
    void surfaceLastHandler(const sensor_msgs::PointCloud2ConstPtr& pointCloudIn) {
        surf_last->clear();
        pcl::fromROSMsg(*pointCloudIn, *surf_last);
        new_surf = true;
    }

    /* subscribe the odometry from lidar odometry： 3Hz */
    void odomHandler(const nav_msgs::Odometry::ConstPtr& odomIn) {

        time_new_odom = odomIn->header.stamp.toSec();
        odom_cur = odomIn;

        new_odom = true;
    }

    /* subscribe the relative odometry (between two scan) from lidar odometry: 10Hz */
    void eachOdomHandler(const nav_msgs::Odometry::ConstPtr& odomIn) {
        each_odom_buf.push_back(odomIn);

        if(each_odom_buf.size() > 5000)
            each_odom_buf[each_odom_buf.size() - 5001] = nullptr;

        new_each_odom = true;
    }

    double gps2utc(double gps_week, double gps_second){
        
        return (gps_week * 604800.0 + gps_second - timeshift_IMUtoGNSS) + 315964800.0;
    }

    void imuHandler(const sensor_msgs::ImuConstPtr& ImuIn) {
        time_last_imu = ImuIn->header.stamp.toSec();

        sensor_msgs::ImuPtr ImuIn_tmp (new sensor_msgs::Imu(*ImuIn));

        imu_buf.push_back(ImuIn_tmp);
        /* imu buff to save imu */
        if(imu_buf.size() > 60000)
            imu_buf[imu_buf.size() - 60001] = nullptr;

        /* current imu time */
        if (cur_time_imu < 0)
            cur_time_imu = time_last_imu;

        if (!first_imu) {
            Eigen::Quaterniond quat(ImuIn->orientation.w,
                                    ImuIn->orientation.x,
                                    ImuIn->orientation.y,
                                    ImuIn->orientation.z);

            Rs[0] = quat.toRotationMatrix();

            abs_poses[0][0] = ImuIn->orientation.w;
            abs_poses[0][1] = ImuIn->orientation.x;
            abs_poses[0][2] = ImuIn->orientation.y;
            abs_poses[0][3] = ImuIn->orientation.z;

            /* set zero orientation */
            if(true)
            {
                // Eigen::Quaterniond quat_(0.9996295, 0.0011795, 0.027192, 0);
                tf2::Quaternion gt_init_q;
                gt_init_q.setRPY(initial_rpy_[0] * 3.1415926/180, initial_rpy_[1] * 3.1415926/180, initial_rpy_[2] * 3.1415926/180);
                gt_init_q.normalize();
                initial_Quat = Eigen::Quaterniond(gt_init_q[3],gt_init_q[0],gt_init_q[1],gt_init_q[2]);
                Rs[0] = initial_Quat.toRotationMatrix();

                abs_poses[0][0] = initial_Quat.w();
                abs_poses[0][1] = initial_Quat.x();
                abs_poses[0][2] = initial_Quat.y();
                abs_poses[0][3] = initial_Quat.z();
            }

            first_imu = true;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            dx = ImuIn->linear_acceleration.x;
            dy = ImuIn->linear_acceleration.y;
            dz = ImuIn->linear_acceleration.z;

            rx = ImuIn->angular_velocity.x;
            ry = ImuIn->angular_velocity.y;
            rz = ImuIn->angular_velocity.z;

            Eigen::Vector3d linear_acceleration(dx, dy, dz);
            Eigen::Vector3d angular_velocity(rx, ry, rz);
            acc_0 = linear_acceleration;
            gyr_0 = angular_velocity;
            pre_integrations.push_back(new Preintegration(acc_0, gyr_0, Bas[0], Bgs[0]));
            pre_integrations.back()->g_vec_ = -g;
        }
    }

    /* transform a point based on the given position */
    void transformPoint(PointType const *const pi, PointType *const po) {
        Eigen::Quaterniond quaternion(abs_pose[0],
                                      abs_pose[1],
                                      abs_pose[2],
                                      abs_pose[3]);
        Eigen::Vector3d transition(abs_pose[4],
                                   abs_pose[5],
                                   abs_pose[6]);

        Eigen::Vector3d ptIn(pi->x, pi->y, pi->z);
        Eigen::Vector3d ptOut = quaternion * ptIn + transition;


        po->x = ptOut.x();
        po->y = ptOut.y();
        po->z = ptOut.z();
        po->intensity = pi->intensity;
    }

    /* transform a point based on the given position */
    void transformPoint(PointType const *const pi, PointType *const po, Eigen::Quaterniond quaternion, Eigen::Vector3d transition) {
        Eigen::Vector3d ptIn(pi->x, pi->y, pi->z);
        Eigen::Vector3d ptOut = quaternion * ptIn + transition;

        po->x = ptOut.x();
        po->y = ptOut.y();
        po->z = ptOut.z();
        po->intensity = pi->intensity;
    }

    /* transform point clouds */
    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn) {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int numPts = cloudIn->points.size();
        cloudOut->resize(numPts);

        for (int i = 0; i < numPts; ++i) {
            PointType ptIn = cloudIn->points[i];
            PointType ptOut;
            transformPoint(&ptIn, &ptOut);
            cloudOut->points[i] = ptOut;
        }
        return cloudOut;
    }

    /* transform point clouds */
    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, PointPoseInfo * PointInfoIn) {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        Eigen::Quaterniond quaternion(PointInfoIn->qw,
                                      PointInfoIn->qx,
                                      PointInfoIn->qy,
                                      PointInfoIn->qz);
        Eigen::Vector3d transition(PointInfoIn->x,
                                   PointInfoIn->y,
                                   PointInfoIn->z);

        int numPts = cloudIn->points.size();
        cloudOut->resize(numPts);

        for (int i = 0; i < numPts; ++i) {
            Eigen::Vector3d ptIn(cloudIn->points[i].x, cloudIn->points[i].y, cloudIn->points[i].z);
            Eigen::Vector3d ptOut = quaternion * ptIn + transition;

            PointType pt;
            pt.x = ptOut.x();
            pt.y = ptOut.y();
            pt.z = ptOut.z();
            pt.intensity = cloudIn->points[i].intensity;

            cloudOut->points[i] = pt;
        }

        return cloudOut;
    }

    /* transform point clouds */
    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, Eigen::Quaterniond quaternion, Eigen::Vector3d transition) {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int numPts = cloudIn->points.size();
        cloudOut->resize(numPts);

        for (int i = 0; i < numPts; ++i) {
            Eigen::Vector3d ptIn(cloudIn->points[i].x, cloudIn->points[i].y, cloudIn->points[i].z);
            Eigen::Vector3d ptOut = quaternion * ptIn + transition;

            PointType pt;
            pt.x = ptOut.x();
            pt.y = ptOut.y();
            pt.z = ptOut.z();
            pt.intensity = cloudIn->points[i].intensity;

            cloudOut->points[i] = pt;
        }

        return cloudOut;
    }

    double pointDistance(PointType p1, PointType p2)
    {
        return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
    }

    double pointDistance(PointPoseInfo p1, PointPoseInfo p2)
    {
        return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
    }

    /* process imu data */
    void processIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity) {
        if(pre_integrations.size() < abs_poses.size()) {
            pre_integrations.push_back(new Preintegration(acc_0, gyr_0, Bas.back(), Bgs.back()));
            pre_integrations.back()->g_vec_ = -g;
            Bas.push_back(Bas.back());
            Bgs.push_back(Bgs.back());
            Rs.push_back(Rs.back());
            Ps.push_back(Ps.back());
            Vs.push_back(Vs.back());
        }

        Eigen::Vector3d un_acc_0 = Rs.back() * (acc_0 - Bas.back()) - g; //
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs.back(); //
        Rs.back() *= deltaQ(un_gyr * dt).toRotationMatrix();
        Eigen::Vector3d un_acc_1 = Rs.back() * (linear_acceleration - Bas.back()) - g; //
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps.back() += dt * Vs.back() + 0.5 * dt * dt * un_acc;
        Vs.back() += dt * un_acc;

        pre_integrations.back()->push_back(dt, linear_acceleration, angular_velocity);

        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    void getLowerUpperIdx(double targetT, int& lower_idx, int&  upper_idx)
    {
        double diff = 10000000;
        /* find the low idx*/
        for (int i = keyframe_idx[keyframe_idx.size()-slide_window_width]; i <= keyframe_idx.back(); i++)
        {
            double t = keyframe_time[i-1];
            double diffTmp = fabs(t - targetT);
            if(diffTmp < diff && t<targetT) 
            {
                lower_idx = i;
                diff = diffTmp;
            }
        }

        diff = 10000000;
        /* find the up idx*/
        for (int i = keyframe_idx[keyframe_idx.size()-slide_window_width]; i <= keyframe_idx.back(); i++)
        {
            double t = keyframe_time[i-1];
            double diffTmp = fabs(t - targetT);
            if(diffTmp < diff && t>targetT) 
            {
                upper_idx = i;
                diff = diffTmp;
            }
        }
    }

    void getGlobalLowerUpperIdx(double targetT, int& lower_idx, int&  upper_idx)
    {
        double diff = 10000000;
        /* find the low idx*/
        for (int i = keyframe_idx[0]; i < pose_info_keyframe_batch->points.size(); i++)
        {
            double t = keyframe_time[i-1];
            double diffTmp = fabs(t - targetT);
            if(diffTmp < diff && t<targetT)
            {
                lower_idx = i;
                diff = diffTmp;
            }
        }

        diff = 10000000;
        /* find the up idx*/
        for (int i = keyframe_idx[0]; i < pose_info_keyframe_batch->points.size(); i++)
        {
            double t = keyframe_time[i-1];
            double diffTmp = fabs(t - targetT);
            if(diffTmp < diff && t>targetT)
            {
                upper_idx = i;
                diff = diffTmp;
            }
        }
    }

    /* get the GNSS measurements within a time period */
    void getGNSSDDMeasurementsWithinPeriod(double windowStartTime, double windowEndTime,
                                         std::vector<nlosExclusion::GNSS_Raw_Array>& user_gnss_meas_vec_tmp,
                                         std::vector<nlosExclusion::GNSS_Raw_Array>& ref_gnss_meas_vec_tmp,
                                         double timeDiff)
    {

        int length = gnss_raw_map.size();
        std::map<int, nlosExclusion::GNSS_Raw_Array>::iterator iter_pr;
        iter_pr = gnss_raw_map.begin();

        for(int i=0; i < length; i++, iter_pr++)
        {
            int gps_time_idx = iter_pr->first;
            double iter_gps_time = double(gps_time_idx/10) + 1600000000.0;

            if(iter_gps_time>=windowStartTime && iter_gps_time<=windowEndTime) {
                std::map<int, nlosExclusion::GNSS_Raw_Array>::iterator iter_ref_pr;
                iter_ref_pr = station_gnss_raw_map.begin();
                int diff = 0, max_diff = fabs(iter_ref_pr->first - gps_time_idx);
                int ref_idx = iter_ref_pr->first;
                for (int j=0; j<station_gnss_raw_map.size(); j++, iter_ref_pr++) {
                    diff = fabs(iter_ref_pr->first - gps_time_idx);
                    if (diff < max_diff) {
                        ref_idx = iter_ref_pr->first;
                        max_diff = diff;
                    }
                }
                if (max_diff < 300) {
                    user_gnss_meas_vec_tmp.push_back(gnss_raw_map[gps_time_idx]);
                    ref_gnss_meas_vec_tmp.push_back(station_gnss_raw_map[ref_idx]);
                }
            }
        }

    }

    /* get alligned GNSS data from reference station and user */
    bool prepareGPSDDPsrData(nlosExclusion::GNSS_Raw_Array& user_gnss_data, nlosExclusion::GNSS_Raw_Array& ref_gnss_data, int& mPrn)
    {
        /* check and filter the data from user and station */
        nlosExclusion::GNSS_Raw_Array refgnssDataTmp;
        nlosExclusion::GNSS_Raw_Array usergnssDataTmp;
        for(int i = 0; i < user_gnss_data.GNSS_Raws.size(); i++)
        {
            int prn = user_gnss_data.GNSS_Raws[i].prn_satellites_index;
            for (int j=0; j<ref_gnss_data.GNSS_Raws.size(); j++) {
                if(prn==ref_gnss_data.GNSS_Raws[j].prn_satellites_index && m_GNSS_Tools.PRNisGPS(prn) && user_gnss_data.GNSS_Raws[i].raw_pseudorange > 1000) // check whether is GPS
                {
                    refgnssDataTmp.GNSS_Raws.push_back(ref_gnss_data.GNSS_Raws[j]);
                    usergnssDataTmp.GNSS_Raws.push_back(user_gnss_data.GNSS_Raws[i]);
                }
            }

        }
        ref_gnss_data  = refgnssDataTmp;
        user_gnss_data = usergnssDataTmp;

        /* get the master satellite id */
        double maxEle = 0;
        for(int m = 0;  m < user_gnss_data.GNSS_Raws.size(); m++) //
        {
            double ele = user_gnss_data.GNSS_Raws[m].elevation;
            if(fabs(ele)>maxEle)
            {
                maxEle = ele;
                mPrn = m;
            }
        }

        return true;
    }
    bool prepareBDSDDPsrData(nlosExclusion::GNSS_Raw_Array& user_gnss_data, nlosExclusion::GNSS_Raw_Array& ref_gnss_data, int& mPrn)
    {
        /* check and filter the data from user and station */
        nlosExclusion::GNSS_Raw_Array refgnssDataTmp;
        nlosExclusion::GNSS_Raw_Array usergnssDataTmp;
        for(int i = 0; i < user_gnss_data.GNSS_Raws.size(); i++)
        {
            int prn = user_gnss_data.GNSS_Raws[i].prn_satellites_index;
            for (int j=0; j<ref_gnss_data.GNSS_Raws.size(); j++) {
                if(prn==ref_gnss_data.GNSS_Raws[j].prn_satellites_index && m_GNSS_Tools.PRNisBeidou(prn) && user_gnss_data.GNSS_Raws[i].raw_pseudorange > 1000) // check whether is BDS
                {
                    refgnssDataTmp.GNSS_Raws.push_back(ref_gnss_data.GNSS_Raws[j]);
                    usergnssDataTmp.GNSS_Raws.push_back(user_gnss_data.GNSS_Raws[i]);
                }
            }

        }
        ref_gnss_data  = refgnssDataTmp;
        user_gnss_data = usergnssDataTmp;

        /* get the master satellite id */
        double maxEle = 0;
        for(int m = 0;  m < user_gnss_data.GNSS_Raws.size(); m++) //
        {
            double ele = user_gnss_data.GNSS_Raws[m].elevation;
            if(fabs(ele)>maxEle)
            {
                maxEle = ele;
                mPrn = m;
            }
        }

        return true;
    }
    bool prepareGALDDPsrData(nlosExclusion::GNSS_Raw_Array& user_gnss_data, nlosExclusion::GNSS_Raw_Array& ref_gnss_data, int& mPrn)
    {
        /* check and filter the data from user and station */
        nlosExclusion::GNSS_Raw_Array refgnssDataTmp;
        nlosExclusion::GNSS_Raw_Array usergnssDataTmp;
        for(int i = 0; i < user_gnss_data.GNSS_Raws.size(); i++)
        {
            int prn = user_gnss_data.GNSS_Raws[i].prn_satellites_index;
            for (int j=0; j<ref_gnss_data.GNSS_Raws.size(); j++) {
                if(prn==ref_gnss_data.GNSS_Raws[j].prn_satellites_index && m_GNSS_Tools.PRNisGAL(prn)) // check whether is BDS
                {
                    refgnssDataTmp.GNSS_Raws.push_back(ref_gnss_data.GNSS_Raws[j]);
                    usergnssDataTmp.GNSS_Raws.push_back(user_gnss_data.GNSS_Raws[i]);
                }
            }

        }
        ref_gnss_data  = refgnssDataTmp;
        user_gnss_data = usergnssDataTmp;

        /* get the master satellite id */
        double maxEle = 0;
        for(int m = 0;  m < user_gnss_data.GNSS_Raws.size(); m++) //
        {
            double ele = user_gnss_data.GNSS_Raws[m].elevation;
            if(fabs(ele)>maxEle)
            {
                maxEle = ele;
                mPrn = m;
            }
        }

        return true;
    }
    bool prepareGLODDPsrData(nlosExclusion::GNSS_Raw_Array& user_gnss_data, nlosExclusion::GNSS_Raw_Array& ref_gnss_data, int& mPrn)
    {
        /* check and filter the data from user and station */
        nlosExclusion::GNSS_Raw_Array refgnssDataTmp;
        nlosExclusion::GNSS_Raw_Array usergnssDataTmp;
        for(int i = 0; i < user_gnss_data.GNSS_Raws.size(); i++)
        {
            int prn = user_gnss_data.GNSS_Raws[i].prn_satellites_index;
            for (int j=0; j<ref_gnss_data.GNSS_Raws.size(); j++) {
                if(prn==ref_gnss_data.GNSS_Raws[j].prn_satellites_index && m_GNSS_Tools.PRNisGLONASS(prn)) // check whether is BDS
                {
                    refgnssDataTmp.GNSS_Raws.push_back(ref_gnss_data.GNSS_Raws[j]);
                    usergnssDataTmp.GNSS_Raws.push_back(user_gnss_data.GNSS_Raws[i]);
                }
            }

        }
        ref_gnss_data  = refgnssDataTmp;
        user_gnss_data = usergnssDataTmp;

        /* get the master satellite id */
        double maxEle = 0;
        for(int m = 0;  m < user_gnss_data.GNSS_Raws.size(); m++) //
        {
            double ele = user_gnss_data.GNSS_Raws[m].elevation;
            if(fabs(ele)>maxEle)
            {
                maxEle = ele;
                mPrn = m;
            }
        }

        return true;
    }

    int getMasterPrn(nlosExclusion::GNSS_Raw_Array gnss_data, int &mIndex)
    {

        /* get the master satellite id */
        double maxEle = 0;
        int mPrn = -1;
        for(int m = 0;  m < gnss_data.GNSS_Raws.size(); m++) //
        {
            double ele = gnss_data.GNSS_Raws[m].elevation;
            if(fabs(ele)>maxEle)
            {
                maxEle = ele;
                mPrn = gnss_data.GNSS_Raws[m].prn_satellites_index;
                mIndex = m;
            }
        }

        return mPrn;
    }

    bool getDMatrix(nlosExclusion::GNSS_Raw_Array gnss_data, int mPrn, MatrixXd& Dmatrix)
    {
        int row = gnss_data.GNSS_Raws.size()-1;
        int col = gnss_data.GNSS_Raws.size();
        Dmatrix.resize(gnss_data.GNSS_Raws.size()-1, gnss_data.GNSS_Raws.size());

        Dmatrix.setZero();
        for(int i = 0; i < row;i++)
            for(int j = 0; j < col; j++)
            {
                if(j==mPrn)
                    Dmatrix(i,j) = 1;

                if(i<mPrn)
                    Dmatrix(i,i) = -1;
                if(i==mPrn)
                    Dmatrix(i,i+1) = -1;
                if(i>mPrn)
                    Dmatrix(i,i+1) = -1;
                // if(j<mPrn)
                // {
                //     Dmatrix(i,i) = -1;
                // }
                // if(j>mPrn)
                // {
                //     Dmatrix(i,i+1) = -1;
                // }
            }
        return true;
    }

    /* add DD pseudorange residual blocks */
    bool addDDPsrResFactor(ceres::Problem& problem, nlosExclusion::GNSS_Raw_Array user_gnss_data, nlosExclusion::GNSS_Raw_Array ref_gnss_data, int mPrn, MatrixXd weight_matrix, int leftKey, int rightKey, double ts_ratio, double DDpsrThreshold)
    {
        Eigen::Vector3d lever_arm_T = ts_ratio < 0.5? Translation_GNSS_IMU(tmpQuat[leftKey]) : Translation_GNSS_IMU(tmpQuat[rightKey]);

        /* add the doppler and DD pseudorange factors */
        dd_psr_factor_20 *dd_psr_factor = new dd_psr_factor_20(user_gnss_data, ref_gnss_data, weight_matrix, mPrn, ts_ratio, lever_arm_T, DDpsrThreshold);
        auto ID = problem.AddResidualBlock(dd_psr_factor, NULL, tmpTrans[leftKey], tmpTrans[rightKey], para_yaw_enu_local, para_anc_ecef);

        return true;
    }

    bool addDDPsrResFactor_gl(ceres::Problem& problem, nlosExclusion::GNSS_Raw_Array user_gnss_data, nlosExclusion::GNSS_Raw_Array ref_gnss_data, int mPrn, MatrixXd weight_matrix, int leftKey, int rightKey, double ts_ratio, double DDpsrThreshold)
    {
        Eigen::Vector3d lever_arm_T = ts_ratio < 0.5? Translation_GNSS_IMU(gl_tmpQuat[leftKey]) : Translation_GNSS_IMU(gl_tmpQuat[rightKey]);
        Eigen::Vector3d Station_pos = Eigen::Vector3d (station_x_, station_y_, station_z_);
        int weight_matrix_size = weight_matrix.cols();
        Eigen::MatrixXd identity_weight_matrix = Eigen::MatrixXd::Identity(weight_matrix_size, weight_matrix_size);

        dd_psr_factor_20 *dd_psr_factor = new dd_psr_factor_20(user_gnss_data, ref_gnss_data, identity_weight_matrix, mPrn, ts_ratio, Station_pos, DDpsrThreshold);
        auto ID = problem.AddResidualBlock(dd_psr_factor, NULL, gl_tmpTrans[leftKey], gl_tmpTrans[rightKey], para_yaw_enu_local, para_anc_ecef); //new ceres::HuberLoss(1.0)

        return true;
    }

    void addGNSSFactor()
    {
        int keyframe_size = pose_keyframe->points.size();
        if (keyframe_size <= slide_window_width) return;

        int proc_kf_idx = keyframe_size - slide_window_width;

        if (GNSSQueue.empty()) {
            return;
        }

        // wait for system initialized and settles down
        if (pose_info_keyframe->points.empty()) {
            return;
        }
        else
        {
            if (pointDistance(last_GNSS_add_pos, pose_info_keyframe->points[proc_kf_idx]) < 5) {
                return;
            }
        }

        // pose covariance small, no need to correct
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold) {
            return;
        }

        // last gps position
        static PointType lastGPSPoint;
        double proc_frame_time = 0;

        proc_frame_time = pose_info_keyframe->points[proc_kf_idx].time + timeshift_IMUtoGNSS;

        while (!GNSSQueue.empty())
        {
            if (GNSSQueue.front().header.stamp.toSec() < proc_frame_time - 0.2)
            {
                GNSSQueue.pop_front();
            }
            else if (GNSSQueue.front().header.stamp.toSec() > proc_frame_time + 0.2)
            {
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = GNSSQueue.front();
                GNSSQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[1];
                float noise_z = thisGPS.pose.covariance[2];
                if (noise_x > gnssCovThreshold || noise_y > gnssCovThreshold) {
                    continue;
                }

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(proc_kf_idx, gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                local_pose_graph.add(gps_factor);

                GNSSAdded = true;
                last_GNSS_add_pos = pose_info_keyframe->points[proc_kf_idx];
                break;
            }
        }
        return;
    }

    void addLIOFactor() {
        int keyframe_size = pose_keyframe->points.size();
        int proc_kf_idx = keyframe_size - slide_window_width;
        //add poses to global graph
        if (keyframe_size == slide_window_width) {
            gtsam::Rot3 rotation = gtsam::Rot3::Quaternion(pose_info_keyframe->points[0].qw,
                                                           pose_info_keyframe->points[0].qx,
                                                           pose_info_keyframe->points[0].qy,
                                                           pose_info_keyframe->points[0].qz);
            gtsam::Point3 transition = gtsam::Point3(pose_keyframe->points[0].x,
                                                     pose_keyframe->points[0].y,
                                                     pose_keyframe->points[0].z);

            // Initialization for global pose graph
            local_pose_graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(rotation, transition), prior_noise));
            local_init_estimate.insert(0, gtsam::Pose3(rotation, transition));
        }

            /* insert all the dense regular frames between two keyframes */
        else if(keyframe_size > slide_window_width) {
            gtsam::Rot3 rotationLast = gtsam::Rot3::Quaternion(pose_info_keyframe->points[proc_kf_idx-1].qw,
                                                               pose_info_keyframe->points[proc_kf_idx-1].qx,
                                                               pose_info_keyframe->points[proc_kf_idx-1].qy,
                                                               pose_info_keyframe->points[proc_kf_idx-1].qz);
            gtsam::Point3 transitionLast = gtsam::Point3(pose_keyframe->points[proc_kf_idx-1].x,
                                                         pose_keyframe->points[proc_kf_idx-1].y,
                                                         pose_keyframe->points[proc_kf_idx-1].z);

            gtsam::Rot3 rotationCur = gtsam::Rot3::Quaternion(pose_info_keyframe->points[proc_kf_idx].qw,
                                                              pose_info_keyframe->points[proc_kf_idx].qx,
                                                              pose_info_keyframe->points[proc_kf_idx].qy,
                                                              pose_info_keyframe->points[proc_kf_idx].qz);
            gtsam::Point3 transitionCur = gtsam::Point3(pose_keyframe->points[proc_kf_idx].x,
                                                        pose_keyframe->points[proc_kf_idx].y,
                                                        pose_keyframe->points[proc_kf_idx].z);
            gtsam::Pose3 poseFrom = gtsam::Pose3(rotationLast, transitionLast);
            gtsam::Pose3 poseTo = gtsam::Pose3(rotationCur, transitionCur);

            local_pose_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(proc_kf_idx - 1,
                                                                    proc_kf_idx,
                                                                    poseFrom.between(poseTo),
                                                                    odom_noise));
            local_init_estimate.insert(proc_kf_idx, poseTo);
        }
    }

    /* 3D LiDAR Aided GNSS sliding window */
    void optimizeSlidingWindowWithLandMark() {
        if(slide_window_width < 1) return;
        if(keyframe_idx.size() < slide_window_width) return;

//        Timer t_optimize_slidingwindow("optimizeSlidingWindowWithLandMark");

        first_opt = true;

        int windowSize = keyframe_idx[keyframe_idx.size()-1] - keyframe_idx[keyframe_idx.size()-slide_window_width] + 1; // always equals to slide_window_width

        kd_tree_surf_local_map->setInputCloud(surf_local_map_ds);
        sensor_msgs::PointCloud2::Ptr sw_clouds_map_ptr(new sensor_msgs::PointCloud2);
        pcl::toROSMsg(*surf_local_map_ds, *sw_clouds_map_ptr);
        sw_clouds_map_ptr->header.frame_id = "GLIO";
        pub_sub_gl_map.publish(*sw_clouds_map_ptr);

        /* GNSS measurements within the sliding window */
        std::vector<std::vector<ObsPtr>> gnss_meas_vec_tmp;
        std::vector<nlosExclusion::GNSS_Raw_Array> user_gnss_meas_vec_tmp, ref_gnss_meas_vec_tmp;
#if 1
        bool new_gnss = false;
        if(gnss_meas_vec.size()>4)
        {
            double windowStartTime = keyframe_time[keyframe_idx[keyframe_idx.size()-slide_window_width]-1];
            double windowEndTime = keyframe_time[keyframe_idx.back()-1];
            getGNSSDDMeasurementsWithinPeriod(windowStartTime, windowEndTime, user_gnss_meas_vec_tmp, ref_gnss_meas_vec_tmp, timeshift_IMUtoGNSS);

            for (int i=0; i<user_gnss_meas_vec_tmp.size(); i++) {
                gtime_t gps_time_ = gpst2time(user_gnss_meas_vec_tmp[i].GNSS_Raws[0].GNSS_week, user_gnss_meas_vec_tmp[i].GNSS_Raws[0].GNSS_time);
                int epoch_time = round((time2sec(gps_time_) - 1600000000 - timeshift_IMUtoGNSS)*10);
                if (!epoch_time_idx.count(epoch_time)) {
                    epoch_time_idx[epoch_time] = epoch_idx;
                    epoch_idx++;
                    new_gnss = true;
                }
            }
        }
#endif
        std::map<int, std::map<int, std::vector<int>>> local_epoch_prn_index;

        //multiple iterations enables the re-searching of correspondance
        int iteration_num = 1;
        double DDpsr_threshold[iteration_num] = {10};
        double psr_threshold[iteration_num] = {10};

        for (int iterCount = 0; iterCount < iteration_num; ++iterCount) {
            ceres::LossFunction *lossFunction = new ceres::HuberLoss(lossKernel);
            ceres::LocalParameterization *quatParameterization = new ceres::QuaternionParameterization();
            ceres::Problem problem;

            /* init vectors for evaluation */
            std::vector<ceres::ResidualBlockId> imuPIIDs;

            //eigen to double: initialize the states
            for (int i = keyframe_idx[keyframe_idx.size()-slide_window_width]; i <= keyframe_idx.back(); i++){

                if (iterCount == 0) {
                    Eigen::Quaterniond tmpQ(Rs[i]);
                    tmpQuat[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][0] =  tmpQ.w();
                    tmpQuat[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][1] =  tmpQ.x();
                    tmpQuat[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][2] =  tmpQ.y();
                    tmpQuat[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][3] =  tmpQ.z();
                    tmpTrans[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][0] = Ps[i][0];
                    tmpTrans[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][1] = Ps[i][1];
                    tmpTrans[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][2] = Ps[i][2];

                    tmp_rcv_dt[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][0] = rcv_dt[i - 1][0];
                    tmp_rcv_dt[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][1] = rcv_dt[i - 1][1];
                    tmp_rcv_dt[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][2] = rcv_dt[i - 1][2];

                    abs_poses[i][0] = tmpQ.w();
                    abs_poses[i][1] = tmpQ.x();
                    abs_poses[i][2] = tmpQ.y();
                    abs_poses[i][3] = tmpQ.z();
                    abs_poses[i][4] = Ps[i][0];
                    abs_poses[i][5] = Ps[i][1];
                    abs_poses[i][6] = Ps[i][2];

                    for (int j = 0; j < 9; j++) {
                        tmpSpeedBias[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][j] = para_speed_bias[i][j];
                    }
                }

                //add lidar parameters
                problem.AddParameterBlock(tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]], 3);
                problem.AddParameterBlock(tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]], 4, quatParameterization);

                //add IMU parameters
                problem.AddParameterBlock(tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]], 9);

                // add rcv dt
                problem.AddParameterBlock(tmp_rcv_dt[i-keyframe_idx[keyframe_idx.size()-slide_window_width]], 3);

                /* initialize the yaw offset parameters (set this as constant) */
                problem.AddParameterBlock(para_yaw_enu_local, 1);
                problem.SetParameterBlockConstant(para_yaw_enu_local);

                /* initialize the anchor point (ENU reference point) parameters */
                problem.AddParameterBlock(para_anc_ecef, 3);
                problem.SetParameterBlockConstant(para_anc_ecef);
            }

            problem.AddParameterBlock(para_rcv_ddt, EPOCH_SIZE);

            abs_pose = abs_poses.back();
#if 1
            /* add the marginalization factor */
            if (last_marginalization_info) {
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                problem.AddResidualBlock(marginalization_factor, NULL,
                                         last_marginalization_parameter_blocks);
            }


            /* prior factor for imu preintegration typically this is not used 
             * this is to increase the stability of IMU bias estimation, this can further be improved by set the upper bound of the IMU bias, refer to the FGO project
            */
            if(!marg) {
                //add prior factor
                for(int i = 0; i < slide_window_width - 1; i++) {

                    vector<double> tmps;
                    for(int j = 0; j < 9; j++) {
                        tmps.push_back(tmpSpeedBias[i][j]);
                    }
                    ceres::CostFunction *speedBiasPriorFactor = SpeedBiasPriorFactorAutoDiff::Create(tmps);
                    problem.AddResidualBlock(speedBiasPriorFactor, NULL, tmpSpeedBias[i]);
                }

            }
#endif

#if 1 // IMU factor
            /* add IMU pre-integration factor */
            double sum_imu_residual = 0;
            for (int idx = keyframe_idx[keyframe_idx.size()-slide_window_width]; idx < keyframe_idx.back(); ++idx) {
                ImuFactor *imuFactor = new ImuFactor(pre_integrations[idx+1]);
                ImuFactor *imuFactor_test = new ImuFactor(pre_integrations[idx+1]);

                auto ID = problem.AddResidualBlock(imuFactor, NULL, tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpSpeedBias[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpTrans[idx+1-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpQuat[idx+1-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpSpeedBias[idx+1-keyframe_idx[keyframe_idx.size()-slide_window_width]]);
            }
#endif

#if 1 // scan to map lidar constraints
            /* add the LiDAR plannar factor */
            double sum_plane_residual = 0;
            for (int idx = keyframe_idx[keyframe_idx.size()-slide_window_width]; idx <= keyframe_idx.back(); idx++) {
                Eigen::Quaterniond Q2 = Eigen::Quaterniond(tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][2],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][3]);
                Eigen::Vector3d T2 = Eigen::Vector3d(tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][2]);

                Eigen::Quaterniond Q2_ = Eigen::Quaterniond(tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][2],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][3]);
                Eigen::Vector3d T2_ = Eigen::Vector3d(tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][2]);


                Q2 = Q2 * q_lb.inverse(); // orientation of imu to local world frame
                T2 = T2 - Q2 * t_lb; // translation of imu to local world frame

                int idVec = idx - keyframe_idx[keyframe_idx.size()-slide_window_width];

                if (surf_local_map_ds->points.size() > 50) {
                    findCorrespondingSurfFeatures(idx-1, Q2, T2);
                    featureSelection(idx-1, Q2_, T2_);

                    /* add plannar feature factorS */
                    for (int i = 0; i < vec_surf_res_cnt[idVec]; ++i) {
                        Eigen::Vector3d currentPt(vec_surf_cur_pts[idVec]->points[i].x,
                                                  vec_surf_cur_pts[idVec]->points[i].y,
                                                  vec_surf_cur_pts[idVec]->points[i].z);
                        Eigen::Vector3d norm(vec_surf_normal[idVec]->points[i].x,
                                             vec_surf_normal[idVec]->points[i].y,
                                             vec_surf_normal[idVec]->points[i].z);
                        double normInverse = vec_surf_normal[idVec]->points[i].intensity;
                        ceres::CostFunction *costFunction = LidarPlaneNormFactor::Create(currentPt, norm, q_lb, t_lb,
                                                                                         normInverse,
                                                                                         vec_surf_scores[idVec][i]);

                        auto ID = problem.AddResidualBlock(costFunction, lossFunction, tmpTrans[idx - keyframe_idx[
                                                                   keyframe_idx.size() - slide_window_width]],
                                                           tmpQuat[idx - keyframe_idx[keyframe_idx.size() -
                                                                                      slide_window_width]]);
                    }
                }
                else {
                    ROS_WARN("Not enough feature points from the map");
                }

            }
#endif

            /* add the GNSS Doppler, DD pseudorange/carrier measurements */
            int gnssSize = user_gnss_meas_vec_tmp.size();


#if 0 // add gnss constraints
            /*dk add gnss constraints*/
            int epochsize = user_gnss_meas_vec_tmp.size();
            vector<vector<double>> tmpIA_org;
            vector<vector<int>> tmpIAInfo;
            for (int i=0; i<epochsize; i++) {
                nlosExclusion::GNSS_Raw_Array gnss_data = user_gnss_meas_vec_tmp[i];
                nlosExclusion::GNSS_Raw_Array ref_gnss_data = ref_gnss_meas_vec_tmp[i];

                /* select the low and upper headers */
                int lower_idx = -1;
                int upper_idx = 10000000;
                gtime_t gps_time_ = gpst2time(gnss_data.GNSS_Raws[0].GNSS_week, gnss_data.GNSS_Raws[0].GNSS_time);
                const double obs_local_ts = time2sec(gps_time_) - timeshift_IMUtoGNSS;
                int cur_epoch_time = round((obs_local_ts - 1600000000)*10);
                if (!epoch_time_idx.count(cur_epoch_time)) {
                    continue;
                }
                if (obs_local_ts > keyframe_time[keyframe_time.size()-1] || obs_local_ts < keyframe_time[0]) continue;

                //  get time by keyframe_time[lower_idx-1]
                getLowerUpperIdx(obs_local_ts, lower_idx, upper_idx);
                const double lower_ts = keyframe_time[lower_idx-1];
                const double upper_ts = keyframe_time[upper_idx-1];

                double ts_ratio = (upper_ts-obs_local_ts) / (upper_ts-lower_ts);

                int leftKey = lower_idx-keyframe_idx[keyframe_idx.size()-slide_window_width];
                int rightKey = upper_idx-keyframe_idx[keyframe_idx.size()-slide_window_width];
                if (leftKey<0 || rightKey<0) continue;

                MatrixXd weight_matrix;
                weight_matrix = m_GNSS_Tools.cofactorMatrixCal_WLS(gnss_data, "WLS"); //goGPS
                weight_matrix = Doppler2PSRWeight * weight_matrix;
                int sv_cnt = gnss_data.GNSS_Raws.size();
                double t = gnss_data.GNSS_Raws[0].GNSS_time;
#if 1
                /* add dopp constraints */
                for(int j =0; j < sv_cnt; j++) {
                    bool sat_active = false;
                    for (int k=0; k<ref_gnss_data.GNSS_Raws.size(); k++) {
                        if (gnss_data.GNSS_Raws[j].prn_satellites_index == ref_gnss_data.GNSS_Raws[k].prn_satellites_index) sat_active = true;
                    }
                    if (!sat_active) continue;
                    std::string sat_sys;
                    if(m_GNSS_Tools.PRNisGPS(gnss_data.GNSS_Raws[j].prn_satellites_index)) {
                        sat_sys = "GPS";
//                        continue;
                    }
                    else if (m_GNSS_Tools.PRNisBeidou(gnss_data.GNSS_Raws[j].prn_satellites_index)){
                        sat_sys = "BDS";
//                        continue;
                    }
                    else if (m_GNSS_Tools.PRNisGAL(gnss_data.GNSS_Raws[j].prn_satellites_index)){
                        sat_sys = "GAL";
                        continue;
                    }
                    else if (m_GNSS_Tools.PRNisGLONASS(gnss_data.GNSS_Raws[j].prn_satellites_index)){
                        sat_sys = "GLO";
                        continue;
                    }

                    Eigen::Vector3d lever_arm_T = ts_ratio < 0.5? Translation_GNSS_IMU(tmpQuat[leftKey]) : Translation_GNSS_IMU(tmpQuat[rightKey]);

                    double sin_yaw_diff = std::sin(para_yaw_enu_local[0]);
                    double cos_yaw_diff = std::cos(para_yaw_enu_local[0]);
                    Eigen::Matrix3d R_enu_local; // rotation of local to enu
                    R_enu_local << cos_yaw_diff, -sin_yaw_diff, 0,
                            sin_yaw_diff,  cos_yaw_diff, 0,
                            0           ,  0           , 1;

                    Eigen::Matrix3d R_ecef_local = R_ecef_enu * R_enu_local; // local to ecef

#if 1 // sf doppler
                    ceres::CostFunction* tcdoppler_function = new ceres::AutoDiffCostFunction
                            <tcdopplerFactor, 1, 3, 9, 3, 9, EPOCH_SIZE, 1, 3>
                            (new tcdopplerFactor(sat_sys, gnss_data.GNSS_Raws[j], ts_ratio, lever_arm_T, R_ecef_local, sqrt(1/weight_matrix(j,j))));

                    tcdoppler_function->Evaluate(parameters_evo, residuals_evo, nullptr);

                    auto ID = problem.AddResidualBlock(tcdoppler_function, new ceres::HuberLoss(1.0),
                                                       tmpTrans[leftKey], tmpSpeedBias[leftKey], tmpTrans[rightKey], tmpSpeedBias[rightKey],
                                                       para_rcv_ddt, para_yaw_enu_local, para_anc_ecef);
#endif

                }
#endif

                /* add dd psr constraints */
                //GPS dd l1
                int mPrn = -1;
                prepareGPSDDPsrData(gnss_data, ref_gnss_data, mPrn);
                Eigen::MatrixXd Dmatrix;
                if(gnss_data.GNSS_Raws.size()>2 && 1)
                {
                    getDMatrix(gnss_data, mPrn, Dmatrix);
                    Eigen::MatrixXd subweight_matrix;
                    subweight_matrix = m_GNSS_Tools.cofactorMatrixCal_WLS(gnss_data, "WLS"); //goGPS cofactorMatrixCal_WLS

                    MatrixXd R_matrix;
                    R_matrix = Dmatrix * subweight_matrix.inverse() * Dmatrix.transpose();
                    R_matrix = R_matrix.cwiseSqrt();
                    R_matrix = R_matrix.inverse();

                    addDDPsrResFactor(problem, gnss_data, ref_gnss_data, mPrn, R_matrix, leftKey, rightKey, ts_ratio, DDpsr_threshold[iterCount]);
                }

                //BDS dd l1
                mPrn = -1;
                gnss_data = user_gnss_meas_vec_tmp[i];
                ref_gnss_data = ref_gnss_meas_vec_tmp[i];
                prepareBDSDDPsrData(gnss_data, ref_gnss_data, mPrn);
                if(gnss_data.GNSS_Raws.size()>2 && 1)
                {
                    getDMatrix(gnss_data, mPrn, Dmatrix);
                    Eigen::MatrixXd subweight_matrix;
                    subweight_matrix = m_GNSS_Tools.cofactorMatrixCal_WLS(gnss_data, "WLS"); //goGPS cofactorMatrixCal_WLS

                    MatrixXd R_matrix;
                    R_matrix = Dmatrix * subweight_matrix.inverse() * Dmatrix.transpose();
                    R_matrix = R_matrix.cwiseSqrt();
                    R_matrix = R_matrix.inverse();

                    addDDPsrResFactor(problem, gnss_data, ref_gnss_data, mPrn, R_matrix, leftKey, rightKey, ts_ratio, DDpsr_threshold[iterCount]);
                }

#if 0
                //GAL dd l1
                mPrn = -1;
                gnss_data = user_gnss_meas_vec_tmp[i];
                ref_gnss_data = ref_gnss_meas_vec_tmp[i];
                prepareGALDDPsrData(gnss_data, ref_gnss_data, mPrn);
                if(gnss_data.GNSS_Raws.size()>2 && 1)
                {
                    getDMatrix(gnss_data, mPrn, Dmatrix);
                    Eigen::MatrixXd subweight_matrix;
                    subweight_matrix = m_GNSS_Tools.cofactorMatrixCal_WLS(gnss_data, "WLS"); //goGPS cofactorMatrixCal_WLS

                    MatrixXd R_matrix;
                    R_matrix = Dmatrix * subweight_matrix.inverse() * Dmatrix.transpose();
                    R_matrix = R_matrix.cwiseSqrt();
                    R_matrix = R_matrix.inverse();

                    addDDPsrResFactor(problem, gnss_data, ref_gnss_data, mPrn, R_matrix, leftKey, rightKey, ts_ratio, DDpsr_threshold[iterCount]);
                }

                //GLO dd l1
                mPrn = -1;
                gnss_data = user_gnss_meas_vec_tmp[i];
                ref_gnss_data = ref_gnss_meas_vec_tmp[i];
                prepareGLODDPsrData(gnss_data, ref_gnss_data, mPrn);
                if(gnss_data.GNSS_Raws.size()>2 && 1)
                {
                    getDMatrix(gnss_data, mPrn, Dmatrix);
                    Eigen::MatrixXd subweight_matrix;
                    subweight_matrix = m_GNSS_Tools.cofactorMatrixCal_WLS(gnss_data, "WLS"); //goGPS cofactorMatrixCal_WLS

                    MatrixXd R_matrix;
                    R_matrix = Dmatrix * subweight_matrix.inverse() * Dmatrix.transpose();
                    R_matrix = R_matrix.cwiseSqrt();
                    R_matrix = R_matrix.inverse();

                    addDDPsrResFactor(problem, gnss_data, ref_gnss_data, mPrn, R_matrix, leftKey, rightKey, ts_ratio, DDpsr_threshold[iterCount]);
                }
#endif
            }
#endif

            /* setup the solver related options */
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // DENSE_QR
            options.num_threads = 1;
            options.max_num_iterations = 15; //max_num_iter
            options.trust_region_strategy_type = ceres::DOGLEG;
            options.minimizer_progress_to_stdout = false;
            options.use_nonmonotonic_steps = false;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            while(para_yaw_enu_local[0] > M_PI)   para_yaw_enu_local[0] -= 2.0*M_PI;
            while(para_yaw_enu_local[0] < -M_PI)  para_yaw_enu_local[0] += 2.0*M_PI;

            /* unify the quaterniond */
            for(int i = 0; i < windowSize; i++) {
                if(tmpQuat[i][0] < 0) {
                    Eigen::Quaterniond tmp(tmpQuat[i][0],
                            tmpQuat[i][1],
                            tmpQuat[i][2],
                            tmpQuat[i][3]);
                    tmp = unifyQuaternion(tmp);
                    tmpQuat[i][0] = tmp.w();
                    tmpQuat[i][1] = tmp.x();
                    tmpQuat[i][2] = tmp.y();
                    tmpQuat[i][3] = tmp.z();
                }
                Eigen::Quaterniond tmp(tmpQuat[i][0],
                                       tmpQuat[i][1],
                                       tmpQuat[i][2],
                                       tmpQuat[i][3]);
                tmp = unifyQuaternion(tmp);
                Eigen::Vector3d sky_point_vect = tmp * Eigen::Vector3d(0, 0, 1);
            }

        }


        MarginalizationInfo *marginalization_info = new MarginalizationInfo();

        if (last_marginalization_info) {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
                if (last_marginalization_parameter_blocks[i] == tmpTrans[0] ||
                        last_marginalization_parameter_blocks[i] == tmpQuat[0] ||
                        last_marginalization_parameter_blocks[i] == tmpSpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);

            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->AddResidualBlockInfo(residual_block_info);
        }

        /* marginalization for the prior imu pre-integration: typically this is not used */
        if(!marg) {
            //add prior factor
            for(int i = 0; i < slide_window_width - 1; i++) {

                vector<double*> tmp;
                tmp.push_back(tmpTrans[i]);
                tmp.push_back(tmpQuat[i]);

                vector<int> drop_set;
                if(i == 0) {
                    drop_set.push_back(0);
                    drop_set.push_back(1);
                }

                vector<double> tmps;
                for(int j = 0; j < 9; j++) {
                    tmps.push_back(tmpSpeedBias[i][j]);
                }

                vector<double*> tmp1;
                tmp1.push_back(tmpSpeedBias[i]);

                vector<int> drop_set1;
                if(i == 0) {
                    drop_set1.push_back(0);
                }
                ceres::CostFunction *speedBiasPriorFactor = SpeedBiasPriorFactorAutoDiff::Create(tmps);
                ResidualBlockInfo *residual_block_info1 = new ResidualBlockInfo(speedBiasPriorFactor, NULL,
                                                                                tmp1,
                                                                                drop_set1);

                marginalization_info->AddResidualBlockInfo(residual_block_info1);
            }

            marg = true;
        }

        //marginalization of imu
        ImuFactor *imuFactor = new ImuFactor(pre_integrations[keyframe_idx[keyframe_idx.size()-slide_window_width]+1]);

        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imuFactor, NULL,
                                                                       vector<double *>{
                                                                               tmpTrans[0],
                                                                               tmpQuat[0],
                                                                               tmpSpeedBias[0],
                                                                               tmpTrans[1],
                                                                               tmpQuat[1],
                                                                               tmpSpeedBias[1]
                                                                       },
                                                                       vector<int>{0, 1, 2});

        marginalization_info->AddResidualBlockInfo(residual_block_info);

#if 1
        //marginalization of lidar factors
        for (int idx = keyframe_idx[keyframe_idx.size()-slide_window_width]; idx <= keyframe_idx.back(); idx++) {
            ceres::LossFunction *lossFunction = new ceres::HuberLoss(lossKernel);
            // ceres::LossFunction *lossFunction = new ceres::CauchyLoss(2.0);
            int idVec = idx - keyframe_idx[keyframe_idx.size()-slide_window_width];
            if (surf_local_map_ds->points.size() > 50) {
                vector<double*> tmp;
                tmp.push_back(tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]]);
                tmp.push_back(tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]]);

                for (int i = 0; i < vec_surf_res_cnt[idVec]; ++i) {
                    Eigen::Vector3d currentPt(vec_surf_cur_pts[idVec]->points[i].x,
                                              vec_surf_cur_pts[idVec]->points[i].y,
                                              vec_surf_cur_pts[idVec]->points[i].z);
                    Eigen::Vector3d norm(vec_surf_normal[idVec]->points[i].x,
                                         vec_surf_normal[idVec]->points[i].y,
                                         vec_surf_normal[idVec]->points[i].z);
                    double normInverse = vec_surf_normal[idVec]->points[i].intensity;

                    //LidarPlaneNormAnalyticFactor *costFunction = new LidarPlaneNormAnalyticFactor(currentPt, norm, normInverse);
                    ceres::CostFunction *costFunction = LidarPlaneNormFactor::Create(currentPt, norm, q_lb, t_lb, normInverse, vec_surf_scores[idVec][i]); //vec_surf_scores[idVec][i] * 1000 / vec_surf_res_cnt[idVec]

                    vector<int> drop_set;
                    if(idx == keyframe_idx[keyframe_idx.size()-slide_window_width]) {
                        drop_set.push_back(0);
                        drop_set.push_back(1);
                    }
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(costFunction, lossFunction,
                                                                                   tmp,
                                                                                   drop_set);
                    marginalization_info->AddResidualBlockInfo(residual_block_info);
                }

            }

            vec_surf_cur_pts[idVec]->clear();
            vec_surf_normal[idVec]->clear();
            vec_surf_res_cnt[idVec] = 0;
            vec_surf_scores[idVec].clear();
        }
#endif

        marginalization_info->PreMarginalize();
        marginalization_info->Marginalize();

#if 1
        /* shift for the states to be optimized */
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i < windowSize; ++i) {
            addr_shift[reinterpret_cast<long>(tmpTrans[i])] = tmpTrans[i-1];
            addr_shift[reinterpret_cast<long>(tmpQuat[i])] = tmpQuat[i-1];
            addr_shift[reinterpret_cast<long>(tmpSpeedBias[i])] = tmpSpeedBias[i-1];

            /* add shift for receiver clock error drift */
//            addr_shift[reinterpret_cast<long>(para_rcv_ddt+i)] = para_rcv_ddt+i-1;

            /* add shift for yaw difference */
            addr_shift[reinterpret_cast<long>(para_yaw_enu_local)] = para_yaw_enu_local;

            /* add shift for anchor point */
            addr_shift[reinterpret_cast<long>(para_anc_ecef)] = para_anc_ecef;
        }

        vector<double *> parameter_blocks = marginalization_info->GetParameterBlocks(addr_shift);


        if (last_marginalization_info) {
            delete last_marginalization_info;
        }
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
#endif

        //double to eigen
        for (int i = keyframe_idx[keyframe_idx.size()-slide_window_width]; i <= keyframe_idx.back(); ++i){
            double dp0 = Ps[i][0] - tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
            double dp1 = Ps[i][1] - tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
            double dp2 = Ps[i][2] - tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            double pnorm = sqrt(dp0*dp0+dp1*dp1+dp2*dp2);

            double dv0 = Vs[i][0] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
            double dv1 = Vs[i][1] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
            double dv2 = Vs[i][2] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            double vnorm = sqrt(dv0*dv0+dv1*dv1+dv2*dv2);

            double dba1 = para_speed_bias[i][3] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3];
            double dba2 = para_speed_bias[i][4] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][4];
            double dba3 = para_speed_bias[i][5] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][5];
            double dbg1 = para_speed_bias[i][6] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][6];
            double dbg2 = para_speed_bias[i][7] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][7];
            double dbg3 = para_speed_bias[i][8] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][8];

            Eigen::Vector3d ba_tmp (tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3],
                                    tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][4],
                                    tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][5]);
            Eigen::Vector3d bg_tmp (tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][6],
                                    tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][7],
                                    tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][8]);

            Eigen::Quaterniond dq = Eigen::Quaterniond (tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                    tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                    tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2],
                    tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3]).normalized().inverse() *
                    Eigen::Quaterniond(Rs[i]);
            double qnorm = dq.vec().norm();

            /* update rcv dt */
            rcv_dt[i-1][0] = tmp_rcv_dt[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
            rcv_dt[i-1][1] = tmp_rcv_dt[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
            rcv_dt[i-1][2] = tmp_rcv_dt[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];


            /* The trans difference between the initialized and optimized state */
            if(pnorm < 100) {
                abs_poses[i][4] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                abs_poses[i][5] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                abs_poses[i][6] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];

                Ps[i][0] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                Ps[i][1] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                Ps[i][2] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            } else {
                ROS_WARN("bad optimization result of p!!!!!!!!!!!!!");
            }

            /* The trans difference between the initialized and optimized state */
            if(qnorm < 10) {
                abs_poses[i][0] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                abs_poses[i][1] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                abs_poses[i][2] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
                abs_poses[i][3] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3];

                Rs[i] = Eigen::Quaterniond (tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2],
                        tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3]).normalized().toRotationMatrix();
            } else
                ROS_WARN("bad optimization result of q!!!!!!!!!!!!!");

            /* The vel difference between the initialized and optimized state */
            if(vnorm < 100) {
                for(int j = 0; j < 3; j++) {
                    para_speed_bias[i][j] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][j];
                }
                Vs[i][0] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                Vs[i][1] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                Vs[i][2] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            } else {
                ROS_WARN("bad optimization result of v!!!!!!!!!!!!!");
            }

            /* The bias of acc difference between the initialized and optimized state */
            if(abs(dba1) < 22) {
                para_speed_bias[i][3] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3];
                Bas[i][0] = para_speed_bias[i][3];
            } else
//                ROS_WARN("bad ba1!!!!!!!!!!");

            if(abs(dba2) < 22) {
                para_speed_bias[i][4] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][4];
                Bas[i][1] = para_speed_bias[i][4];
            } else
//                ROS_WARN("bad ba2!!!!!!!!!!");

            if(abs(dba3) < 22) {
                para_speed_bias[i][5] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][5];
                Bas[i][2] = para_speed_bias[i][5];
            } else
//                ROS_WARN("bad ba3!!!!!!!!!!");

            if(abs(dbg1) < 22) {
                para_speed_bias[i][6] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][6];
                Bgs[i][0] = para_speed_bias[i][6];
            } else
//                ROS_WARN("bad bg1!!!!!!!!!!");

            if(abs(dbg2) < 22) {
                para_speed_bias[i][7] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][7];
                Bgs[i][1] = para_speed_bias[i][7];
            } else
//                ROS_WARN("bad bg2!!!!!!!!!!");

            if(abs(dbg3) < 22) {
                para_speed_bias[i][8] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][8];
                Bgs[i][2] = para_speed_bias[i][8];
            }
//            else
//                ROS_WARN("bad bg3!!!!!!!!!!");

        }

        new_gnss = false;

        updatePose();

        /* construct batch lidar feature association */
        batchFeatureAssociation();

//        t_optimize_slidingwindow.tic_toc();
    }

    /* 3D LiDAR Aided GNSS batch optimization */
    void optimizeBatchWithLandMark() {
        if (pose_info_keyframe->points.size() < 30) return;
        if (enable_batch_fusion) {
            if ((pose_info_keyframe->points.size() - batch_activate_idx) < 10) {
                return;
            }
            else {
                batch_activate_idx = pose_info_keyframe->points.size();
            }
        }

//        Timer t_optimize_batch("optimizeBatchWithLandMark");
        *pose_info_keyframe_batch = *pose_info_keyframe; //pose of each frame
        int cur_batch_size = pose_info_keyframe_batch->points.size();

        gl_tmpQuat = new double *[cur_batch_size]; // orientation
        gl_tmpTrans = new double *[cur_batch_size]; // translation
        gl_tmpSpeedBias = new double *[cur_batch_size];  // speed, bias_a, bias_g
        start_idx = 0;

        /* generate global map for LiDAR matching */
        surf_global_map.reset(new pcl::PointCloud<PointType>());
        surf_global_map_ds.reset(new pcl::PointCloud<PointType>());

        //multiple iterations enables the re-searching of correspondence
        int iteration_num = 4;
        double DDpsr_threshold[iteration_num] = {1000000000, 10, 8, 6, 4, 3, 2.5}; // 1000000000, 10, 8, 6, 4, 3, 2.5

        for (int iterCount = 0; iterCount < iteration_num; iterCount++) { // 1
            ceres::LossFunction *lossFunction = NULL; //new ceres::HuberLoss(lossKernel)
            ceres::LocalParameterization *quatParameterization = new ceres::QuaternionParameterization();
            ceres::Problem problem;

            std::vector<std::vector<ObsPtr>> gl_gnss_meas_vec_tmp;

            /* GNSS measurements within the sliding window */
            std::vector<nlosExclusion::GNSS_Raw_Array> user_gnss_meas_vec_tmp, ref_gnss_meas_vec_tmp;
            if(gnss_meas_vec.size()>4)
            {
                double gl_windowStartTime = keyframe_time[start_idx];
                double gl_windowEndTime = pose_info_keyframe_batch->points[cur_batch_size-1].time;

                getGNSSDDMeasurementsWithinPeriod(gl_windowStartTime, gl_windowEndTime, user_gnss_meas_vec_tmp, ref_gnss_meas_vec_tmp, timeshift_IMUtoGNSS);

                gl_tmp_rcv_dt = new double *[user_gnss_meas_vec_tmp.size()];
                for (int i=0; i<user_gnss_meas_vec_tmp.size(); i++) {
                    gl_tmp_rcv_dt[i] = new double[3];
                    gl_tmp_rcv_dt[i][0] = 0;
                    gl_tmp_rcv_dt[i][1] = 0;
                    gl_tmp_rcv_dt[i][2] = 0;
                    problem.AddParameterBlock(gl_tmp_rcv_dt[i], 3);
                }
            }

            //eigen to double: initialize the states
            for (int i = start_idx; i < cur_batch_size; i++){

                gl_tmpQuat[i] = new double[4]; // w, x, y, z
                gl_tmpTrans[i] = new double[3]; // x, y, z
                gl_tmpSpeedBias[i] = new double[9]; // V,BA,BG

                Eigen::Quaterniond tmpQ(Rs[i]);
                // gt init
                gl_tmpQuat[i][0]  = pose_info_keyframe_batch->points[i].qw;
                gl_tmpQuat[i][1]  = pose_info_keyframe_batch->points[i].qx;
                gl_tmpQuat[i][2]  = pose_info_keyframe_batch->points[i].qy;
                gl_tmpQuat[i][3]  = pose_info_keyframe_batch->points[i].qz;
                gl_tmpTrans[i][0] = pose_info_keyframe_batch->points[i].x;
                gl_tmpTrans[i][1] = pose_info_keyframe_batch->points[i].y;
                gl_tmpTrans[i][2] = pose_info_keyframe_batch->points[i].z;

                for (int j=0; j<9; j++) {
                    gl_tmpSpeedBias[i][j] = para_speed_bias[i+1][j];
                }

                //add lidar parameters
                problem.AddParameterBlock(gl_tmpTrans[i], 3);
                problem.AddParameterBlock(gl_tmpQuat[i], 4, quatParameterization);

                //add IMU parameters
                problem.AddParameterBlock(gl_tmpSpeedBias[i], 9);

            }
            /* initialize the yaw offset parameters (set this as constant) */
            problem.AddParameterBlock(para_yaw_enu_local, 1);
            problem.SetParameterBlockConstant(para_yaw_enu_local);

            /* initialize the anchor point (ENU reference point) parameters */
            problem.AddParameterBlock(para_anc_ecef, 3);
            problem.SetParameterBlockConstant(para_anc_ecef);

            // relative attitude constraint
            for (int i = start_idx; i < cur_batch_size; i++) {
                Eigen::Quaterniond qi(pose_info_keyframe->points[i].qw,
                                      pose_info_keyframe->points[i].qx,
                                      pose_info_keyframe->points[i].qy,
                                      pose_info_keyframe->points[i].qz);
                qi = unifyQuaternion(qi);

                Eigen::Vector3d pi = Eigen::Vector3d(pose_info_keyframe->points[i].x,
                                                     pose_info_keyframe->points[i].y,
                                                     pose_info_keyframe->points[i].z);
                Eigen::Vector3d p_tmp = pi;
                int factor_count = 0;
                //backward attitude constraint
                for (int j=i; j>=start_idx; j-=1) {
                    if (factor_count == search_range) {
                        factor_count = 0;
                        break;
                    }
                    if (j == i) continue;
                    Eigen::Quaterniond qj(pose_info_keyframe->points[j].qw,
                                          pose_info_keyframe->points[j].qx,
                                          pose_info_keyframe->points[j].qy,
                                          pose_info_keyframe->points[j].qz);
                    Eigen::Vector3d pj = Eigen::Vector3d(pose_info_keyframe->points[j].x,
                                                         pose_info_keyframe->points[j].y,
                                                         pose_info_keyframe->points[j].z);
                    double p_range = (p_tmp - pj).norm();
                    if (p_range > 5/search_range) { //6
                        p_tmp = pj;
                        Eigen::Quaterniond const_diff = qi.inverse() * qj;
                        ceres::CostFunction* delta_q_factor = new ceres::AutoDiffCostFunction<delta_q_factor_auto,3,4,4>
                                (new delta_q_factor_auto(const_diff));
                        auto ID = problem.AddResidualBlock(delta_q_factor, NULL, gl_tmpQuat[i], gl_tmpQuat[j]);
                        factor_count++;
                    }
                }
                //forward attitude constraint
                for (int j=i; j<cur_batch_size; j++) {
                    if (factor_count == search_range) {
                        factor_count = 0;
                        break;
                    }
                    if (j == i) continue;
                    Eigen::Quaterniond qj(pose_info_keyframe->points[j].qw,
                                          pose_info_keyframe->points[j].qx,
                                          pose_info_keyframe->points[j].qy,
                                          pose_info_keyframe->points[j].qz);
                    Eigen::Vector3d pj = Eigen::Vector3d(pose_info_keyframe->points[j].x,
                                                         pose_info_keyframe->points[j].y,
                                                         pose_info_keyframe->points[j].z);
                    double p_range = (p_tmp - pj).norm();
                    if (p_range > 5/search_range) { // 6
                        p_tmp = pj;
                        Eigen::Quaterniond const_diff = qi.inverse() * qj;
                        ceres::CostFunction* delta_q_factor = new ceres::AutoDiffCostFunction<delta_q_factor_auto,3,4,4>
                                (new delta_q_factor_auto(const_diff));
                        auto ID = problem.AddResidualBlock(delta_q_factor, NULL, gl_tmpQuat[i], gl_tmpQuat[j]);
                        factor_count++;
                    }
                }
            }

            /* add parameter block for receiver clock drift */
            problem.AddParameterBlock(gl_para_rcv_ddt, EPOCH_SIZE);

            /* add scan-to-multiscan factor */
            if (sms_fusion_level == 0) {
                /* add relative pose factor */
                //forward backward batch constraint
                for (int idx = start_idx + search_range; idx < cur_batch_size; idx++) {
                    for (int ms_i = 1; ms_i < search_range; ms_i++) {
                        Eigen::Vector3d tmpTrans = Eigen::Vector3d(pose_info_keyframe->points[idx].x,
                                                                   pose_info_keyframe->points[idx].y,
                                                                   pose_info_keyframe->points[idx].z) -
                                                   Eigen::Vector3d(pose_info_keyframe->points[idx - ms_i].x,
                                                                   pose_info_keyframe->points[idx - ms_i].y,
                                                                   pose_info_keyframe->points[idx - ms_i].z);
                        tmpTrans = Eigen::Quaterniond(pose_info_keyframe->points[idx - ms_i].qw,
                                                      pose_info_keyframe->points[idx - ms_i].qx,
                                                      pose_info_keyframe->points[idx - ms_i].qy,
                                                      pose_info_keyframe->points[idx - ms_i].qz).inverse() * tmpTrans;

                        Eigen::Quaterniond tmpQuat = Eigen::Quaterniond(pose_info_keyframe->points[idx - ms_i].qw,
                                                                        pose_info_keyframe->points[idx - ms_i].qx,
                                                                        pose_info_keyframe->points[idx - ms_i].qy,
                                                                        pose_info_keyframe->points[idx -
                                                                                                   ms_i].qz).inverse() *
                                                     Eigen::Quaterniond(pose_info_keyframe->points[idx].qw,
                                                                        pose_info_keyframe->points[idx].qx,
                                                                        pose_info_keyframe->points[idx].qy,
                                                                        pose_info_keyframe->points[idx].qz);
                        ceres::CostFunction *relativePose_factor = LidarPoseFactorBatchRelativeAutoDiff::Create(tmpQuat,
                                                                                                                tmpTrans);
                        problem.AddResidualBlock(relativePose_factor, NULL, gl_tmpTrans[idx - ms_i],
                                                 gl_tmpQuat[idx - ms_i], gl_tmpTrans[idx], gl_tmpQuat[idx]);
                    }
                }
                for (int idx = start_idx; idx < cur_batch_size - search_range; idx++) {
                    for (int ms_i = 1; ms_i < search_range; ms_i++) {
                        Eigen::Vector3d tmpTrans = Eigen::Vector3d(pose_info_keyframe->points[idx + ms_i].x,
                                                                   pose_info_keyframe->points[idx + ms_i].y,
                                                                   pose_info_keyframe->points[idx + ms_i].z) -
                                                   Eigen::Vector3d(pose_info_keyframe->points[idx].x,
                                                                   pose_info_keyframe->points[idx].y,
                                                                   pose_info_keyframe->points[idx].z);
                        tmpTrans = Eigen::Quaterniond(pose_info_keyframe->points[idx].qw,
                                                      pose_info_keyframe->points[idx].qx,
                                                      pose_info_keyframe->points[idx].qy,
                                                      pose_info_keyframe->points[idx].qz).inverse() * tmpTrans;

                        Eigen::Quaterniond tmpQuat = Eigen::Quaterniond(pose_info_keyframe->points[idx].qw,
                                                                        pose_info_keyframe->points[idx].qx,
                                                                        pose_info_keyframe->points[idx].qy,
                                                                        pose_info_keyframe->points[idx].qz).inverse() *
                                                     Eigen::Quaterniond(pose_info_keyframe->points[idx + ms_i].qw,
                                                                        pose_info_keyframe->points[idx + ms_i].qx,
                                                                        pose_info_keyframe->points[idx + ms_i].qy,
                                                                        pose_info_keyframe->points[idx + ms_i].qz);
                        ceres::CostFunction *relativePose_factor = LidarPoseFactorBatchRelativeAutoDiff::Create(tmpQuat,
                                                                                                                tmpTrans);
                        problem.AddResidualBlock(relativePose_factor, NULL, gl_tmpTrans[idx], gl_tmpQuat[idx],
                                                 gl_tmpTrans[idx + ms_i], gl_tmpQuat[idx + ms_i]);
                    }
                }
                // vehicle static constraint
                if (0) {
                    int static_keyframe_idx = start_idx;
                    for (int idx = start_idx; idx < cur_batch_size; idx++) {
                        if (idx == static_keyframe_idx) continue;
                        Eigen::Vector3d tmpTrans = Eigen::Vector3d(pose_info_keyframe->points[idx].x,
                                                                   pose_info_keyframe->points[idx].y,
                                                                   pose_info_keyframe->points[idx].z) -
                                                   Eigen::Vector3d(pose_info_keyframe->points[static_keyframe_idx].x,
                                                                   pose_info_keyframe->points[static_keyframe_idx].y,
                                                                   pose_info_keyframe->points[static_keyframe_idx].z);
                        tmpTrans = Eigen::Quaterniond(pose_info_keyframe->points[static_keyframe_idx].qw,
                                                      pose_info_keyframe->points[static_keyframe_idx].qx,
                                                      pose_info_keyframe->points[static_keyframe_idx].qy,
                                                      pose_info_keyframe->points[static_keyframe_idx].qz).inverse() *
                                   tmpTrans;
                        if (tmpTrans.norm() < 0.05) {
                            Eigen::Quaterniond tmpQuat =
                                    Eigen::Quaterniond(pose_info_keyframe->points[static_keyframe_idx].qw,
                                                       pose_info_keyframe->points[static_keyframe_idx].qx,
                                                       pose_info_keyframe->points[static_keyframe_idx].qy,
                                                       pose_info_keyframe->points[static_keyframe_idx].qz).inverse() *
                                    Eigen::Quaterniond(pose_info_keyframe->points[idx].qw,
                                                       pose_info_keyframe->points[idx].qx,
                                                       pose_info_keyframe->points[idx].qy,
                                                       pose_info_keyframe->points[idx].qz);
                            ceres::CostFunction *relativePose_factor = LidarPoseFactorBatchRelativeAutoDiff::Create(
                                    tmpQuat, tmpTrans);
                            problem.AddResidualBlock(relativePose_factor, NULL, gl_tmpTrans[static_keyframe_idx],
                                                     gl_tmpQuat[static_keyframe_idx], gl_tmpTrans[idx],
                                                     gl_tmpQuat[idx]);
                        } else static_keyframe_idx = start_idx;
                    }
                }
            }
            else if (sms_fusion_level == 1) {

                /* add IMU pre-integration factor */
                for (int idx = start_idx; idx < cur_batch_size-1; ++idx) {
                    ImuFactor *imuFactor = new ImuFactor(pre_integrations[idx+1]);
                    auto ID = problem.AddResidualBlock(imuFactor, NULL, gl_tmpTrans[idx],
                                                       gl_tmpQuat[idx],
                                                       gl_tmpSpeedBias[idx],
                                                       gl_tmpTrans[idx+1],
                                                       gl_tmpQuat[idx+1],
                                                       gl_tmpSpeedBias[idx+1]);
                }

                /* add the LiDAR plannar factor */
                for (int idx = start_idx; idx < cur_batch_size; idx++) {
                    int idVec = idx; // start from 0

                    // start of search keyframes
                    int search_idx_start;
                    if (idx >= search_range + start_idx && idx < cur_batch_size - 1 - search_range) {
                        search_idx_start = idx - search_range;
                    }
                    else if (idx < search_range + start_idx) {
                        search_idx_start = start_idx;
                    }
                    else if (idx >= cur_batch_size - 1 - search_range) {
                        search_idx_start = cur_batch_size - 2*search_range - 1;
                    }
                    if (idx > cur_batch_size - 1 - search_range || idx < search_range + start_idx ) {
                        for (int j = search_idx_start; j <= search_idx_start + 2*search_range; j++) {
                            pcl::PointCloud<PointType>::Ptr tmpSurfCurrent(new pcl::PointCloud<PointType>());
                            gl_vec_surf_cur_pts_startend[idx][j] = tmpSurfCurrent;
                            gl_vec_surf_res_cnt_startend[idx][j] = 0;
                            vector<double> tmpD;
                            gl_vec_surf_scores_startend[idx][j] = tmpD;
//                            vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> tmp_pnc;
                            vector<vector<double>> tmp_pnc;
                            gl_vec_surf_normals_cents_startend[idx][j] = tmp_pnc;
                        }
                        findGlobalCorrespondingSurfFeatures_Batch(idx, search_idx_start);
                        globalFeatureSelection_Batch(idx, search_idx_start);
                        /* add plannar feature factorS */
                        Eigen::Quaterniond q_(1., 0., 0., 0.);
                        Eigen::Vector3d t_(0., 0., 0.);
                        for (int search_idx = search_idx_start; search_idx <= search_idx_start + 2*search_range; search_idx++) {
                            if (search_idx == idx) continue;
                            for (int i = 0; i < gl_vec_surf_res_cnt_startend[idVec][search_idx]; ++i) {
                                Eigen::Vector3d currentPt(gl_vec_surf_cur_pts_startend[idVec][search_idx]->points[i].x,
                                                          gl_vec_surf_cur_pts_startend[idVec][search_idx]->points[i].y,
                                                          gl_vec_surf_cur_pts_startend[idVec][search_idx]->points[i].z);
                                Eigen::Matrix<double, 6, 1> norm_cent;
                                norm_cent << gl_vec_surf_normals_cents_startend[idVec][search_idx][i][0],
                                        gl_vec_surf_normals_cents_startend[idVec][search_idx][i][1],
                                        gl_vec_surf_normals_cents_startend[idVec][search_idx][i][2],
                                        gl_vec_surf_normals_cents_startend[idVec][search_idx][i][3],
                                        gl_vec_surf_normals_cents_startend[idVec][search_idx][i][4],
                                        gl_vec_surf_normals_cents_startend[idVec][search_idx][i][5];

                                ceres::CostFunction *costFunction = BinaryLidarPlaneNormFactor::Create(currentPt, norm_cent, gl_vec_surf_scores_startend[idVec][search_idx][i]);

                                auto ID = problem.AddResidualBlock(costFunction, lossFunction, gl_tmpTrans[idx], gl_tmpQuat[idx], gl_tmpTrans[search_idx], gl_tmpQuat[search_idx]);
                            }
                        }
                    }
                    else {
                        /* add plannar feature factorS */
                        Eigen::Quaterniond q_(1., 0., 0., 0.);
                        Eigen::Vector3d t_(0., 0., 0.);
                        for (int search_idx = search_idx_start; search_idx <= search_idx_start + 2*search_range; search_idx++) {
                            if (search_idx == idx) continue;
                            for (int i = 0; i < gl_vec_surf_res_cnt[idVec][search_idx]; ++i) {
                                Eigen::Vector3d currentPt(gl_vec_surf_cur_pts[idVec][search_idx]->points[i].x,
                                                          gl_vec_surf_cur_pts[idVec][search_idx]->points[i].y,
                                                          gl_vec_surf_cur_pts[idVec][search_idx]->points[i].z);
                                Eigen::Matrix<double, 6, 1> norm_cent;
                                norm_cent << gl_vec_surf_normals_cents[idVec][search_idx][i][0],
                                        gl_vec_surf_normals_cents[idVec][search_idx][i][1],
                                        gl_vec_surf_normals_cents[idVec][search_idx][i][2],
                                        gl_vec_surf_normals_cents[idVec][search_idx][i][3],
                                        gl_vec_surf_normals_cents[idVec][search_idx][i][4],
                                        gl_vec_surf_normals_cents[idVec][search_idx][i][5];
                                ceres::CostFunction *costFunction = BinaryLidarPlaneNormFactor::Create(currentPt, norm_cent, gl_vec_surf_scores[idVec][search_idx][i]); //vec_surf_scores[idVec][i] * 1000 / vec_surf_res_cnt[idVec]
                                auto ID = problem.AddResidualBlock(costFunction, lossFunction, gl_tmpTrans[idx], gl_tmpQuat[idx], gl_tmpTrans[search_idx], gl_tmpQuat[search_idx]); // lossFunction
                            }
                        }
                    }
                }
            }

            /* add the GNSS Doppler, DD pseudorange/carrier measurements */
            int gl_gnssSize = gl_gnss_meas_vec_tmp.size();

            int gnssSize = gl_gnss_meas_vec_tmp.size();
            int epochsize = user_gnss_meas_vec_tmp.size();
            Eigen::Vector3d P_add_idx = Eigen::Vector3d(0, 0, 0);

            for (int i=0; i<epochsize; i++) {
                nlosExclusion::GNSS_Raw_Array gnss_data = user_gnss_meas_vec_tmp[i];
                nlosExclusion::GNSS_Raw_Array ref_gnss_data = ref_gnss_meas_vec_tmp[i];

                /* select the low and upper headers */
                int lower_idx = -1;
                int upper_idx = 10000000;
                gtime_t gps_time_ = gpst2time(gnss_data.GNSS_Raws[0].GNSS_week, gnss_data.GNSS_Raws[0].GNSS_time);
                const double obs_local_ts = time2sec(gps_time_) - timeshift_IMUtoGNSS;
                int cur_epoch_time = round((obs_local_ts - 1600000000)*10);
                if (!epoch_time_idx.count(cur_epoch_time)) {
                    cout << "problem cur_epoch_time: " << cur_epoch_time << " obs_local_ts: " << obs_local_ts << " GNSS_week: " << gnss_data.GNSS_Raws[0].GNSS_week << " GNSS_time: " << gnss_data.GNSS_Raws[0].GNSS_time << endl;
                    continue;
                }
                if (obs_local_ts > keyframe_time[keyframe_time.size()-1] || obs_local_ts < keyframe_time[0]) continue;

                //  get time by keyframe_time[lower_idx-1]
                getGlobalLowerUpperIdx(obs_local_ts, lower_idx, upper_idx);
                if ((lower_idx < 0 || lower_idx >= pose_info_keyframe_batch->points.size()) ||
                (upper_idx < 0 || upper_idx >= pose_info_keyframe_batch->points.size())) continue;
                const double lower_ts = keyframe_time[lower_idx-1];
                const double upper_ts = keyframe_time[upper_idx-1];

                double ts_ratio = (upper_ts-obs_local_ts) / (upper_ts-lower_ts);

                int leftKey = lower_idx-1;
                int rightKey = upper_idx-1;

                Eigen::Vector3d Pi(gl_tmpTrans[leftKey][0], gl_tmpTrans[leftKey][1], gl_tmpTrans[leftKey][2]);
                Eigen::Vector3d Pj(gl_tmpTrans[rightKey][0], gl_tmpTrans[rightKey][1], gl_tmpTrans[rightKey][2]);
                Eigen::Vector3d Vi(gl_tmpSpeedBias[leftKey][0], gl_tmpSpeedBias[leftKey][1], gl_tmpSpeedBias[leftKey][2]);
                Eigen::Vector3d Vj(gl_tmpSpeedBias[rightKey][0], gl_tmpSpeedBias[rightKey][1], gl_tmpSpeedBias[rightKey][2]);
                Eigen::Quaterniond Qi(gl_tmpQuat[leftKey][0], gl_tmpQuat[leftKey][1], gl_tmpQuat[leftKey][2], gl_tmpQuat[leftKey][3]);
                Qi = unifyQuaternion(Qi);

                if ((Pj - P_add_idx).norm() < 1) {
                    continue;
                }
                else {
                    P_add_idx = Pj;
                }

                Eigen::Vector3d v_local = ts_ratio * Vi + (1.0 - ts_ratio) * Vj;
                double v_scale = v_local.norm();
                double yaw_diff = para_yaw_enu_local[0]; // yaw difference between enu and local
                double sin_yaw_diff = std::sin(yaw_diff);
                double cos_yaw_diff = std::cos(yaw_diff);
                Eigen::Matrix3d R_enu_local; // rotation of local to enu
                R_enu_local << cos_yaw_diff, -sin_yaw_diff, 0,
                        sin_yaw_diff, cos_yaw_diff, 0,
                        0, 0, 1;
                Eigen::Matrix3d R_ecef_local = R_ecef_enu * R_enu_local; // local to ecef

                const Eigen::Vector3d local_pos_ = ts_ratio * Pi + (1.0 - ts_ratio) * Pj; // itepolation

                Eigen::Vector3d lever_arm_T = ts_ratio < 0.5? Translation_GNSS_IMU(gl_tmpQuat[leftKey]) : Translation_GNSS_IMU(gl_tmpQuat[rightKey]);
                const Eigen::Vector3d local_pos = local_pos_ + lever_arm_T;    // lever arm correction

                Eigen::Vector3d P_ecef = R_ecef_local * local_pos + anc_ecef; // pose in ecef

#if 0 // add doppler constraints
                MatrixXd weight_matrix; //goGPS weighting
                weight_matrix = m_GNSS_Tools.cofactorMatrixCal_WLS(gnss_data, "WLS"); //goGPS
                weight_matrix = Doppler2PSRWeight * weight_matrix;
                int sv_cnt = gnss_data.GNSS_Raws.size();
                double t = gnss_data.GNSS_Raws[0].GNSS_time;

                /* add dopp constraints */
                double avg_dopp = 0; double sNum = 0; double avg_speed = sqrt(gl_tmpSpeedBias[leftKey][0]*gl_tmpSpeedBias[leftKey][0] + gl_tmpSpeedBias[leftKey][1]*gl_tmpSpeedBias[leftKey][1] + gl_tmpSpeedBias[leftKey][2]*gl_tmpSpeedBias[leftKey][2]);

                for(int j =0; j < sv_cnt; j++) {
                    std::string sat_sys;
                    if(m_GNSS_Tools.PRNisGPS(gnss_data.GNSS_Raws[j].prn_satellites_index)) {
                        sat_sys = "GPS";
//                        continue;
                    }
                    else if (m_GNSS_Tools.PRNisBeidou(gnss_data.GNSS_Raws[j].prn_satellites_index)){
                        sat_sys = "BDS";
//                        continue;
                    }
                    else if (m_GNSS_Tools.PRNisGAL(gnss_data.GNSS_Raws[j].prn_satellites_index)){
                        sat_sys = "GAL";
//                        continue;
                    }
                    else if (m_GNSS_Tools.PRNisGLONASS(gnss_data.GNSS_Raws[j].prn_satellites_index)){
                        sat_sys = "GLO";
//                        continue;
                    }

#if 1 // sf doppler
                    ceres::CostFunction* tcdoppler_function = new ceres::AutoDiffCostFunction
                            <tcdopplerFactor, 1, 3, 9, 3, 9, EPOCH_SIZE, 1, 3>
                            (new tcdopplerFactor(i, sat_sys, gnss_data.GNSS_Raws[j], ts_ratio, lever_arm_T, R_ecef_local, sqrt(1/weight_matrix(j,j))));

                    tcdoppler_function->Evaluate(parameters_evo, residuals_evo, nullptr);

                    auto ID = problem.AddResidualBlock(tcdoppler_function, new ceres::HuberLoss(1.0),
                                                       gl_tmpTrans[leftKey], gl_tmpSpeedBias[leftKey], gl_tmpTrans[rightKey], gl_tmpSpeedBias[rightKey],
                                                       gl_para_rcv_ddt, para_yaw_enu_local, para_anc_ecef);
#endif
                }
                avg_dopp /= sNum;

                //add constant rcv ddt factor
                if (i > 0) {
                    ceres::CostFunction* constRcvDdt_function = new ceres::AutoDiffCostFunction
                            <constantClockDriftFactor, 1, EPOCH_SIZE> (new constantClockDriftFactor(i-1, i));
                    problem.AddResidualBlock(constRcvDdt_function, new ceres::HuberLoss(1.0), gl_para_rcv_ddt);
                }
#endif

                /* add dd psr constraints */
                //GPS dd l1
                int mPrn = -1;
                prepareGPSDDPsrData(gnss_data, ref_gnss_data, mPrn);
                Eigen::MatrixXd Dmatrix;
                if(gnss_data.GNSS_Raws.size()>2 && 1)
                {
                    getDMatrix(gnss_data, mPrn, Dmatrix);
                    Eigen::MatrixXd subweight_matrix;
                    subweight_matrix = m_GNSS_Tools.cofactorMatrixCal_WLS(gnss_data, "WLS"); //goGPS cofactorMatrixCal_WLS

                    MatrixXd R_matrix;
                    R_matrix = Dmatrix * subweight_matrix.inverse() * Dmatrix.transpose();
                    R_matrix = R_matrix.cwiseSqrt();
                    R_matrix = R_matrix.inverse();

                    addDDPsrResFactor_gl(problem, gnss_data, ref_gnss_data, mPrn, R_matrix, leftKey, rightKey, ts_ratio, DDpsr_threshold[iterCount]);
                }

                //BDS DD L1
                mPrn = -1;
                gnss_data = user_gnss_meas_vec_tmp[i];
                ref_gnss_data = ref_gnss_meas_vec_tmp[i];
                prepareBDSDDPsrData(gnss_data, ref_gnss_data, mPrn);
                if(gnss_data.GNSS_Raws.size()>2 && 1)
                {
                    getDMatrix(gnss_data, mPrn, Dmatrix);
                    Eigen::MatrixXd subweight_matrix;
                    subweight_matrix = m_GNSS_Tools.cofactorMatrixCal_WLS(gnss_data, "WLS"); //goGPS cofactorMatrixCal_WLS

                    MatrixXd R_matrix;
                    R_matrix = Dmatrix * subweight_matrix.inverse() * Dmatrix.transpose();
                    R_matrix = R_matrix.cwiseSqrt();
                    R_matrix = R_matrix.inverse();

                    addDDPsrResFactor_gl(problem, gnss_data, ref_gnss_data, mPrn, R_matrix, leftKey, rightKey, ts_ratio, DDpsr_threshold[iterCount]);
                }

                //GLO DD L1
                mPrn = -1;
                gnss_data = user_gnss_meas_vec_tmp[i];
                ref_gnss_data = ref_gnss_meas_vec_tmp[i];
                prepareGLODDPsrData(gnss_data, ref_gnss_data, mPrn);
                if(gnss_data.GNSS_Raws.size()>2 && 1)
                {
                    getDMatrix(gnss_data, mPrn, Dmatrix);
                    Eigen::MatrixXd subweight_matrix;
                    subweight_matrix = m_GNSS_Tools.cofactorMatrixCal_WLS(gnss_data, "WLS"); //goGPS cofactorMatrixCal_WLS

                    MatrixXd R_matrix;
                    R_matrix = Dmatrix * subweight_matrix.inverse() * Dmatrix.transpose();
                    R_matrix = R_matrix.cwiseSqrt();
                    R_matrix = R_matrix.inverse();

                    addDDPsrResFactor_gl(problem, gnss_data, ref_gnss_data, mPrn, R_matrix, leftKey, rightKey, ts_ratio, DDpsr_threshold[iterCount]);
                }

                //GAL DD L1
                mPrn = -1;
                gnss_data = user_gnss_meas_vec_tmp[i];
                ref_gnss_data = ref_gnss_meas_vec_tmp[i];
                prepareGALDDPsrData(gnss_data, ref_gnss_data, mPrn);
                if(gnss_data.GNSS_Raws.size()>2 && 1)
                {
                    getDMatrix(gnss_data, mPrn, Dmatrix);
                    Eigen::MatrixXd subweight_matrix;
                    subweight_matrix = m_GNSS_Tools.cofactorMatrixCal_WLS(gnss_data, "WLS"); //goGPS cofactorMatrixCal_WLS

                    MatrixXd R_matrix;
                    R_matrix = Dmatrix * subweight_matrix.inverse() * Dmatrix.transpose();
                    R_matrix = R_matrix.cwiseSqrt();
                    R_matrix = R_matrix.inverse();

                    addDDPsrResFactor_gl(problem, gnss_data, ref_gnss_data, mPrn, R_matrix, leftKey, rightKey, ts_ratio, DDpsr_threshold[iterCount]);
                }
            }

            /* setup the solver related options */
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // DENSE_QR
            options.num_threads = 1;
            options.max_num_iterations = max_num_iter;
            options.trust_region_strategy_type = ceres::TrustRegionStrategyType::DOGLEG;
            options.dogleg_type = ceres::DoglegType::SUBSPACE_DOGLEG;
            options.use_nonmonotonic_steps = true;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            while(para_yaw_enu_local[0] > M_PI)   para_yaw_enu_local[0] -= 2.0*M_PI;
            while(para_yaw_enu_local[0] < -M_PI)  para_yaw_enu_local[0] += 2.0*M_PI;

            /* unify the quaterniond */
            for(int i = start_idx; i < cur_batch_size; i++) {
                if(gl_tmpQuat[i][0] < 0) {
                    Eigen::Quaterniond tmp(gl_tmpQuat[i][0],
                                           gl_tmpQuat[i][1],
                                           gl_tmpQuat[i][2],
                                           gl_tmpQuat[i][3]);
                    tmp = unifyQuaternion(tmp);
                    gl_tmpQuat[i][0] = tmp.w();
                    gl_tmpQuat[i][1] = tmp.x();
                    gl_tmpQuat[i][2] = tmp.y();
                    gl_tmpQuat[i][3] = tmp.z();
                }
            }

            //double to eigen
            for (int i = start_idx; i < cur_batch_size; ++i){
                pose_info_keyframe_batch->points[i].x = gl_tmpTrans[i][0];
                pose_info_keyframe_batch->points[i].y = gl_tmpTrans[i][1];
                pose_info_keyframe_batch->points[i].z = gl_tmpTrans[i][2];
                pose_info_keyframe_batch->points[i].qw = gl_tmpQuat[i][0];
                pose_info_keyframe_batch->points[i].qx = gl_tmpQuat[i][1];
                pose_info_keyframe_batch->points[i].qy = gl_tmpQuat[i][2];
                pose_info_keyframe_batch->points[i].qz = gl_tmpQuat[i][3];

//                for (int j=0; j<9; j++) {
//                    para_speed_bias[i+1][j] = gl_tmpSpeedBias[i][j];
//                }
            }

#if 1 //pub batch trajectory
            if (iterCount == iteration_num - 1) {
                nav_msgs::Path path_batch;
                path_batch.header.frame_id = frame_id;
                for (int i = keyframe_idx[start_idx]; i < cur_batch_size; ++i) {
                    geometry_msgs::PoseStamped batch_pose_msg;
                    batch_pose_msg.header = path_batch.header;
                    batch_pose_msg.pose.orientation.w = pose_info_keyframe_batch->points[i - 1].qw;
                    batch_pose_msg.pose.orientation.x = pose_info_keyframe_batch->points[i - 1].qx;
                    batch_pose_msg.pose.orientation.y = pose_info_keyframe_batch->points[i - 1].qy;
                    batch_pose_msg.pose.orientation.z = pose_info_keyframe_batch->points[i - 1].qz;
                    batch_pose_msg.pose.position.x = pose_info_keyframe_batch->points[i - 1].x;
                    batch_pose_msg.pose.position.y = pose_info_keyframe_batch->points[i - 1].y;
                    batch_pose_msg.pose.position.z = pose_info_keyframe_batch->points[i - 1].z;
                    path_batch.poses.push_back(batch_pose_msg);
                }
                pub_batch_path.publish(path_batch);

#if 1 /* write final result of pose*/
//                batch_result_path_evo = result_path + std::to_string(iterCount) + "GLIO_batch_enu_q_evo.csv";
//                std::ofstream batch_res_output_evo(batch_result_path_evo, std::ios::out);
//                batch_res_output_evo.close();
//                batch_result_path = result_path + std::to_string(iterCount) + "GLIO_batch_enu.csv";
                batch_result_path = result_path + "tc_batch_result.csv";
                std::ofstream batch_res_output_ws(batch_result_path, std::ios::out);
                batch_res_output_ws.close();

                for (int i = keyframe_idx[0]; i < cur_batch_size; ++i) {
                    /* results for evo */
//                    ofstream fout_batch_evo(batch_result_path_evo, ios::app); //GROUND_TRUTH_PATH_EVO
//                    fout_batch_evo.setf(ios::fixed, ios::floatfield);
//                    fout_batch_evo.precision(8);
//                    fout_batch_evo << keyframe_time[i - 1] << ' ';
//                    fout_batch_evo << pose_info_keyframe_batch->points[i - 1].x << ' '
//                                   << pose_info_keyframe_batch->points[i - 1].y << ' '
//                                   << pose_info_keyframe_batch->points[i - 1].z << ' '
//                                   << pose_info_keyframe_batch->points[i-1].qx << ' '
//                                   << pose_info_keyframe_batch->points[i-1].qy << ' '
//                                   << pose_info_keyframe_batch->points[i-1].qz << ' '
//                                   << pose_info_keyframe_batch->points[i-1].qw << '\n';
//                    fout_batch_evo.close();

                    /* write result to file */
                    Eigen::Vector3d batch_Ps_i = Eigen::Vector3d(pose_info_keyframe_batch->points[i - 1].x,
                                                                 pose_info_keyframe_batch->points[i - 1].y,
                                                                 pose_info_keyframe_batch->points[i - 1].z);
                    enu_pos = batch_Ps_i;
                    Eigen::Matrix3d batch_Rs_i = Eigen::Quaterniond (pose_info_keyframe_batch->points[i-1].qw,
                                                                     pose_info_keyframe_batch->points[i-1].qx,
                                                                     pose_info_keyframe_batch->points[i-1].qy,
                                                                     pose_info_keyframe_batch->points[i-1].qz).normalized().toRotationMatrix();
                    enu_ypr = Utility::R2ypr(batch_Rs_i);
                    ecef_pos = anc_ecef + R_ecef_enu * enu_pos; // from enu to ecef
                    Eigen::Vector3d lla_pos = ecef2geo(ecef_pos);

                    ofstream tc_batch_output(batch_result_path, ios::app);
                    tc_batch_output.setf(ios::fixed, ios::floatfield);
                    tc_batch_output.precision(8);
                    const double gnss_ts = keyframe_time[i - 1] + timeshift_IMUtoGNSS;
                    gtime_t gtime = sec2time(gnss_ts);
                    uint32_t gps_week = 0;
                    double gps_sec = time2gpst(gtime, &gps_week);
                    tc_batch_output << keyframe_time[i - 1] << ',';
                    tc_batch_output << gps_week << ',';
                    tc_batch_output << gps_sec << ',';

                    tc_batch_output << lla_pos.x() << ','
                                      << lla_pos.y() << ','
                                      << lla_pos.z() << ','
                                      << enu_ypr.x() << ','
                                      << enu_ypr.y() << ','
                                      << enu_ypr.z() << ','
                                      << enu_pos[0] << ','
                                      << enu_pos[1] << ','
                                      << enu_pos[2] << '\n';
                    tc_batch_output.close();
                }
//                cout << "---------------- Write traj of iteration "<< iterCount << " for evo" << endl;
#endif
            }
#endif

        }

        gl_vec_surf_cur_pts_startend.clear();
        gl_vec_surf_res_cnt_startend.clear();
        gl_vec_surf_scores_startend.clear();
        gl_vec_surf_normals_cents_startend.clear();

//        t_optimize_batch.tic_toc();

    }

    /* construct batch lidar feature association */
    void batchFeatureAssociation() {
        int idx = keyframe_idx.size() - search_range - 1;
        if (keyframe_idx.size() < 2*search_range || idx < search_range) return;
        int search_idx_start = idx - search_range;

        for (int j = search_idx_start; j <= search_idx_start + 2*search_range; j++) {
            pcl::PointCloud<PointType>::Ptr tmpSurfCurrent(new pcl::PointCloud<PointType>());
            gl_vec_surf_cur_pts[idx][j] = tmpSurfCurrent;
            gl_vec_surf_res_cnt[idx][j] = 0;
            vector<double> tmpD;
            gl_vec_surf_scores[idx][j] = tmpD;
//            vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> tmp_pnc;
            vector<vector<double>> tmp_pnc;
            gl_vec_surf_normals_cents[idx][j] = tmp_pnc;
        }
        findGlobalCorrespondingSurfFeaturesAdd_Batch(idx, search_idx_start);
        globalFeatureSelectionAdd_Batch(idx, search_idx_start);

        return;
    }

    /* update the pose */
    void updatePose() {
        abs_pose = abs_poses.back();
        for (int i = keyframe_idx[keyframe_idx.size()-slide_window_width]; i <= keyframe_idx[keyframe_idx.size()-1]; ++i){
            pose_keyframe->points[i-1].x = abs_poses[i][4];
            pose_keyframe->points[i-1].y = abs_poses[i][5];
            pose_keyframe->points[i-1].z = abs_poses[i][6];

            pose_info_keyframe->points[i-1].x = abs_poses[i][4];
            pose_info_keyframe->points[i-1].y = abs_poses[i][5];
            pose_info_keyframe->points[i-1].z = abs_poses[i][6];
            pose_info_keyframe->points[i-1].qw = abs_poses[i][0];
            pose_info_keyframe->points[i-1].qx = abs_poses[i][1];
            pose_info_keyframe->points[i-1].qy = abs_poses[i][2];
            pose_info_keyframe->points[i-1].qz = abs_poses[i][3];
        }
    }

    void optimizeLocalGraph(vector<double*> paraEach) {
        ceres::LocalParameterization *quatParameterization = new ceres::QuaternionParameterization();
        ceres::Problem problem;

        int numPara = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width] - keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1;
        if(numPara==0) return;

        double dQuat[numPara][4];
        double dTrans[numPara][3];

        for(int i = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1] + 1;
            i < keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width]; i++) {
            dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][0] = pose_each_frame->points[i].x;
            dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][1] = pose_each_frame->points[i].y;
            dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][2] = pose_each_frame->points[i].z;

            dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][0] = pose_info_each_frame->points[i].qw;
            dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][1] = pose_info_each_frame->points[i].qx;
            dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][2] = pose_info_each_frame->points[i].qy;
            dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][3] = pose_info_each_frame->points[i].qz;

            problem.AddParameterBlock(dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1], 3);
            problem.AddParameterBlock(dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1], 4, quatParameterization);
        }



        ceres::CostFunction *LeftFactor = LidarPoseLeftFactorAutoDiff::Create(Eigen::Quaterniond(paraEach[1][0], paraEach[1][1], paraEach[1][2], paraEach[1][3]),
                Eigen::Vector3d(paraEach[0][0], paraEach[0][1], paraEach[0][2]),
                Eigen::Quaterniond(pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width - 1].qw,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width - 1].qx,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width - 1].qy,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width - 1].qz),
                Eigen::Vector3d(pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width - 1].x,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width - 1].y,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width - 1].z));
        problem.AddResidualBlock(LeftFactor, NULL, dTrans[0], dQuat[0]);
        for(int i = 0; i < numPara - 1; i++) {
            ceres::CostFunction *Factor = LidarPoseFactorAutoDiff::Create(Eigen::Quaterniond(paraEach[2*i+1][0], paraEach[2*i+1][1], paraEach[2*i+1][2], paraEach[2*i+1][3]),
                    Eigen::Vector3d(paraEach[2*i][0], paraEach[2*i][1], paraEach[2*i][2]));
            problem.AddResidualBlock(Factor, NULL, dTrans[i], dQuat[i], dTrans[i+1], dQuat[i+1]);
        }

        ceres::CostFunction *RightFactor = LidarPoseRightFactorAutoDiff::Create(Eigen::Quaterniond(paraEach.back()[0], paraEach.back()[1], paraEach.back()[2], paraEach.back()[3]),
                Eigen::Vector3d(paraEach[paraEach.size()-2][0], paraEach[paraEach.size()-2][1], paraEach[paraEach.size()-2][2]),
                Eigen::Quaterniond(pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width].qw,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width].qx,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width].qy,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width].qz),
                Eigen::Vector3d(pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width].x,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width].y,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width].z));
        problem.AddResidualBlock(RightFactor, NULL, dTrans[numPara-1], dQuat[numPara-1]);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.max_num_iterations = 15;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        for(int i = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1] + 1;
            i < keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width]; i++) {
            pose_each_frame->points[i].x = dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][0];
            pose_each_frame->points[i].y = dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][1];
            pose_each_frame->points[i].z = dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][2];

            pose_info_each_frame->points[i].x = dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][0];
            pose_info_each_frame->points[i].y = dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][1];
            pose_info_each_frame->points[i].z = dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][2];
            pose_info_each_frame->points[i].qw = dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][0];
            pose_info_each_frame->points[i].qx = dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][1];
            pose_info_each_frame->points[i].qy = dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][2];
            pose_info_each_frame->points[i].qz = dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][3];
        }
    }

    void buildLocalMapWithLandMark() {
        // Initialization
        if (pose_keyframe->points.size() < 1) {
            PointPoseInfo Tbl;
            Tbl.qw = q_bl.w();
            Tbl.qx = q_bl.x();
            Tbl.qy = q_bl.y();
            Tbl.qz = q_bl.z();
            Tbl.x = t_bl.x();
            Tbl.y = t_bl.y();
            Tbl.z = t_bl.z();
            //ROS_INFO("Initialization for local map building");
            *surf_local_map += *transformCloud(surf_last, &Tbl);
            return;
        }

        if (recent_surf_keyframes.size() < local_map_width) {
            recent_surf_keyframes.clear();

            for (int i = pose_keyframe->points.size() - 1; i >= 0; --i) {
                if ((int)pose_keyframe->points[i].intensity < 0) continue;
                if (pose_keyframe->points.size() > local_map_width && i <= pose_keyframe->points.size() - local_map_width) break;
                int idx = (int)pose_keyframe->points[i].intensity;

                Eigen::Quaterniond q_po(pose_info_keyframe->points[idx].qw,
                                        pose_info_keyframe->points[idx].qx,
                                        pose_info_keyframe->points[idx].qy,
                                        pose_info_keyframe->points[idx].qz);

                Eigen::Vector3d t_po(pose_info_keyframe->points[idx].x,
                                     pose_info_keyframe->points[idx].y,
                                     pose_info_keyframe->points[idx].z);

                Eigen::Quaterniond q_tmp = q_po * q_bl;
                Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

                PointPoseInfo Ttmp;
                Ttmp.qw = q_tmp.w();
                Ttmp.qx = q_tmp.x();
                Ttmp.qy = q_tmp.y();
                Ttmp.qz = q_tmp.z();
                Ttmp.x = t_tmp.x();
                Ttmp.y = t_tmp.y();
                Ttmp.z = t_tmp.z();

                recent_surf_keyframes.push_front(transformCloud(surf_frames[idx], &Ttmp));

                if (recent_surf_keyframes.size() >= local_map_width)
                    break;
            }
        }
        // If already more then 50 frames, pop the frames at the beginning
        else {
            if (latest_frame_idx != pose_keyframe->points.size() - 1) {
                recent_surf_keyframes.pop_front();
                latest_frame_idx = pose_keyframe->points.size() - 1;

                Eigen::Quaterniond q_po(pose_info_keyframe->points[latest_frame_idx].qw,
                                        pose_info_keyframe->points[latest_frame_idx].qx,
                                        pose_info_keyframe->points[latest_frame_idx].qy,
                                        pose_info_keyframe->points[latest_frame_idx].qz);

                Eigen::Vector3d t_po(pose_info_keyframe->points[latest_frame_idx].x,
                                     pose_info_keyframe->points[latest_frame_idx].y,
                                     pose_info_keyframe->points[latest_frame_idx].z);

                Eigen::Quaterniond q_tmp = q_po * q_bl;
                Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

                PointPoseInfo Ttmp;
                Ttmp.qw = q_tmp.w();
                Ttmp.qx = q_tmp.x();
                Ttmp.qy = q_tmp.y();
                Ttmp.qz = q_tmp.z();
                Ttmp.x = t_tmp.x();
                Ttmp.y = t_tmp.y();
                Ttmp.z = t_tmp.z();

                recent_surf_keyframes.push_back(transformCloud(surf_frames[latest_frame_idx], &Ttmp));

            }
        }

        surf_local_map->points.clear();
        for (int i = 0; i < recent_surf_keyframes.size(); ++i) {
            *surf_local_map += *recent_surf_keyframes[i];
        }
    }

    void downSampleCloud() {

        ds_filter_surf_map.setInputCloud(surf_local_map);
        ds_filter_surf_map.filter(*surf_local_map_ds);

        pcl::PointCloud<PointType>::Ptr fullDS(new pcl::PointCloud<PointType>());
        ds_filter_surf_map.setInputCloud(full_cloud);
        ds_filter_surf_map.filter(*fullDS);
//        full_clouds_ds.push_back(fullDS);

        surf_last_ds->clear();
        ds_filter_surf.setInputCloud(surf_last);
        ds_filter_surf.filter(*surf_last_ds);
    }

    void findCorrespondingSurfFeatures(int idx, Eigen::Quaterniond q, Eigen::Vector3d t) {
//        Timer t_feature_association("findCorrespondingSurfFeatures");

        double nearst_dist = 0; int count_ = 0;
        bool fCSF = false;
        int idVec = idx - keyframe_idx[keyframe_idx.size()-slide_window_width] + 1;
        vec_surf_res_cnt[idVec] = 0;
        int fail_max_radius = 0;
        int fail_plane_fit = 0;
        int fail_weight = 0;
        for (int i = 0; i < surf_frames[idx]->points.size(); ++i) {
            pt_in_local = surf_frames[idx]->points[i];

            transformPoint(&pt_in_local, &pt_in_map, q, t);
            kd_tree_surf_local_map->nearestKSearch(pt_in_map, 5, pt_search_idx, pt_search_sq_dists);

            Eigen::Matrix<double, 5, 3> matA0 = Eigen::Matrix<double, 5, 3>::Ones();
            Eigen::Matrix<double, 5, 1> matB0 = - Eigen::Matrix<double, 5, 1>::Ones();
            if (pt_search_sq_dists[4] < kd_max_radius) { // last one lasgest
                nearst_dist += fabs(pt_search_sq_dists[4]);
                count_++;
                for (int j = 0; j < 5; ++j) {
                    matA0(j, 0) = surf_local_map_ds->points[pt_search_idx[j]].x;
                    matA0(j, 1) = surf_local_map_ds->points[pt_search_idx[j]].y;
                    matA0(j, 2) = surf_local_map_ds->points[pt_search_idx[j]].z;
                }

                // Get the norm of the plane using linear solver based on QR composition
                Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                double normInverse = 1 / norm.norm();
                norm.normalize(); // get the unit norm

                // Make sure that the plan is fit
                bool planeValid = true;
                for (int j = 0; j < 5; ++j) {
                    if (fabs(norm.x() * surf_local_map_ds->points[pt_search_idx[j]].x +
                             norm.y() * surf_local_map_ds->points[pt_search_idx[j]].y +
                             norm.z() * surf_local_map_ds->points[pt_search_idx[j]].z + normInverse) > surf_dist_thres) {
                        planeValid = false;
                        break;
                    }
                }

                // if one eigenvalue is significantly larger than the other two
                if (planeValid) {
                    float pd = norm.x() * pt_in_map.x + norm.y() * pt_in_map.y + norm.z() *pt_in_map.z + normInverse;
                    float weight = 1 - 0.9 * fabs(pd) / sqrt(sqrt(pt_in_map.x * pt_in_map.x + pt_in_map.y * pt_in_map.y + pt_in_map.z * pt_in_map.z));

                    if(weight > 0.3) {
                        PointType normal;
                        normal.x = weight * norm.x();
                        normal.y = weight * norm.y();
                        normal.z = weight * norm.z();
                        normal.intensity = weight * normInverse;

                        vec_surf_cur_pts[idVec]->push_back(pt_in_local);
                        vec_surf_normal[idVec]->push_back(normal);

                        ++vec_surf_res_cnt[idVec];
                        vec_surf_scores[idVec].push_back(lidar_const*weight);
                        fCSF = true;
                    }
                    else {
                        fail_weight++;
                    }
                }
                else {
                    fail_plane_fit++;
                }
            }
            else {
                fail_max_radius++;
            }
        }
//        t_feature_association.tic_toc();
    }

    void findGlobalCorrespondingSurfFeatures_Batch(int idx, int search_idx_start) {
//        Timer t_feature_association("findGlobalCorrespondingSurfFeatures_Batch");
        int idVec = idx;
        pcl::PointCloud<PointType>::Ptr surf_local_cur_frame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surf_global_cur_frame_map(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surf_global_oth_frame_map(new pcl::PointCloud<PointType>());
        *surf_global_cur_frame_map = *transformCloud(surf_frames[idx], &pose_info_keyframe->points[idx]);
        *surf_local_cur_frame = *surf_frames[idx];

        for (int search_idx = search_idx_start; search_idx <= search_idx_start + 2*search_range; search_idx++) {
            if (search_idx == idx) continue;

            pcl::PointCloud<PointType>::Ptr surf_local_search_frame(new pcl::PointCloud<PointType>());
            *surf_local_search_frame = *surf_frames[search_idx];

            *surf_global_oth_frame_map = *transformCloud(surf_local_search_frame, &pose_info_keyframe->points[search_idx]);
            pcl::PointCloud<PointType>::Ptr tmpSurfCurrent(new pcl::PointCloud<PointType>());
            gl_vec_surf_cur_pts_startend[idx][search_idx] = tmpSurfCurrent;

            pcl::KdTreeFLANN<PointType>::Ptr kd_tree_surf_local_map_batch;
            kd_tree_surf_local_map_batch.reset(new pcl::KdTreeFLANN<PointType>());
            kd_tree_surf_local_map_batch->setInputCloud(surf_global_oth_frame_map);

            int sfi_size = surf_local_cur_frame->points.size();
            int sgf_map_size = surf_global_cur_frame_map->points.size();

            for (int i = 0; i < surf_global_cur_frame_map->points.size(); i++) {
                int idxx = i;
                PointType pt_in_local_;
                double sf_idx_i_x = surf_local_cur_frame->points[i].x;
                pt_in_local_ = surf_local_cur_frame->points[i];
                PointType pt_in_gl_map_;
                pt_in_gl_map_ = surf_global_cur_frame_map->points[i];
                vector<double> normal_cent(6, 0);
                vector<int> pt_search_idx_batch;
                vector<float> pt_search_sq_dists_batch;
                kd_tree_surf_local_map_batch->nearestKSearch(pt_in_gl_map_, 5, pt_search_idx_batch, pt_search_sq_dists_batch);
                Eigen::Matrix<double, 5, 3> matA0 = Eigen::Matrix<double, 5, 3>::Zero();
                Eigen::Matrix<double, 5, 3> matA0_local = Eigen::Matrix<double, 5, 3>::Zero();
                Eigen::Matrix<double, 5, 1> matB0 = - Eigen::Matrix<double, 5, 1>::Ones();
                Eigen::Matrix<double, 5, 1> matB0_local = - Eigen::Matrix<double, 5, 1>::Ones();
                if (pt_search_sq_dists_batch[4] < 1.5) { // last one lasgest
                    double cent_x = 0; double cent_y = 0; double cent_z = 0;
                    for (int j = 0; j < 5; ++j) {
                        matA0(j, 0) = surf_global_oth_frame_map->points[pt_search_idx_batch[j]].x;
                        matA0(j, 1) = surf_global_oth_frame_map->points[pt_search_idx_batch[j]].y;
                        matA0(j, 2) = surf_global_oth_frame_map->points[pt_search_idx_batch[j]].z;
                        matA0_local(j, 0) = surf_local_search_frame->points[pt_search_idx_batch[j]].x;
                        matA0_local(j, 1) = surf_local_search_frame->points[pt_search_idx_batch[j]].y;
                        matA0_local(j, 2) = surf_local_search_frame->points[pt_search_idx_batch[j]].z;
                        cent_x += matA0_local(j, 0);
                        cent_y += matA0_local(j, 1);
                        cent_z += matA0_local(j, 2);
                    }
                    normal_cent[3] = cent_x/5.;
                    normal_cent[4] = cent_y/5.;
                    normal_cent[5] = cent_z/5.;
                    // Get the norm of the plane using linear solver based on QR composition
                    Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                    double normInverse = 1 / norm.norm();
                    norm.normalize(); // get the unit norm
                    Eigen::Vector3d norm_local = matA0_local.colPivHouseholderQr().solve(matB0_local);
                    norm_local.normalize(); // get the unit norm
                    // Make sure that the plan is fit
                    bool planeValid = true;
                    for (int j = 0; j < 5; ++j) {
                        if (fabs(norm.x() * surf_global_oth_frame_map->points[pt_search_idx_batch[j]].x +
                                 norm.y() * surf_global_oth_frame_map->points[pt_search_idx_batch[j]].y +
                                 norm.z() * surf_global_oth_frame_map->points[pt_search_idx_batch[j]].z + normInverse) > 0.18) {
                            planeValid = false;
                            break;
                        }
                    }
                    // if one eigenvalue is significantly larger than the other two
                    if (planeValid) {
                        float pd = norm.x() * pt_in_gl_map_.x + norm.y() * pt_in_gl_map_.y + norm.z() *pt_in_gl_map_.z + normInverse;
                        float weight = 1 - 0.9 * fabs(pd) / sqrt(sqrt(pt_in_gl_map_.x * pt_in_gl_map_.x + pt_in_gl_map_.y * pt_in_gl_map_.y + pt_in_gl_map_.z * pt_in_gl_map_.z));
                        if(weight > 0.3) {
                            PointType normal;
                            normal.x = weight * norm.x();
                            normal.y = weight * norm.y();
                            normal.z = weight * norm.z();
                            normal.intensity = weight * normInverse;
                            normal_cent[0] = norm_local.x();
                            normal_cent[1] = norm_local.y();
                            normal_cent[2] = norm_local.z();
                            gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.push_back(pt_in_local_);
                            ++gl_vec_surf_res_cnt_startend[idVec][search_idx];
                            gl_vec_surf_scores_startend[idVec][search_idx].push_back(2.5*weight);
                            gl_vec_surf_normals_cents_startend[idVec][search_idx].push_back(normal_cent);
                        }
                    }
                }
            }
        }
//        t_feature_association.tic_toc();
    }

    void findGlobalCorrespondingSurfFeaturesAdd_Batch(int idx, int search_idx_start) {
//        Timer t_feature_association("findGlobalCorrespondingSurfFeaturesAdd_Batch");
        int idVec = idx;
        pcl::PointCloud<PointType>::Ptr surf_global_cur_frame_map(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surf_global_oth_frame_map(new pcl::PointCloud<PointType>());
        *surf_global_cur_frame_map = *transformCloud(surf_frames[idx], &pose_info_keyframe->points[idx]);
        for (int search_idx = search_idx_start; search_idx <= search_idx_start + 2*search_range; search_idx++) {
            if (search_idx == idx) continue;

            *surf_global_oth_frame_map = *transformCloud(surf_frames[search_idx], &pose_info_keyframe->points[search_idx]);
            pcl::PointCloud<PointType>::Ptr tmpSurfCurrent(new pcl::PointCloud<PointType>());
            gl_vec_surf_cur_pts[idVec][search_idx] = tmpSurfCurrent;

            pcl::KdTreeFLANN<PointType>::Ptr kd_tree_surf_local_map_batch;
            kd_tree_surf_local_map_batch.reset(new pcl::KdTreeFLANN<PointType>());
            kd_tree_surf_local_map_batch->setInputCloud(surf_global_oth_frame_map);
            for (int i = 0; i < surf_global_cur_frame_map->points.size(); ++i) {
                PointType pt_in_local_;
                pt_in_local_ = surf_frames[idx]->points[i];
                PointType pt_in_gl_map_;
                pt_in_gl_map_ = surf_global_cur_frame_map->points[i];
                vector<double> normal_cent(6, 0);
                vector<int> pt_search_idx_batch;
                vector<float> pt_search_sq_dists_batch;
                kd_tree_surf_local_map_batch->nearestKSearch(pt_in_gl_map_, 5, pt_search_idx_batch, pt_search_sq_dists_batch);
                Eigen::Matrix<double, 5, 3> matA0 = Eigen::Matrix<double, 5, 3>::Zero();
                Eigen::Matrix<double, 5, 3> matA0_local = Eigen::Matrix<double, 5, 3>::Zero();
                Eigen::Matrix<double, 5, 1> matB0 = - Eigen::Matrix<double, 5, 1>::Ones();
                Eigen::Matrix<double, 5, 1> matB0_local = - Eigen::Matrix<double, 5, 1>::Ones();
                if (pt_search_sq_dists_batch[4] < 1.5) { // last one lasgest
                    double cent_x = 0; double cent_y = 0; double cent_z = 0;
                    for (int j = 0; j < 5; ++j) {
                        matA0(j, 0) = surf_global_oth_frame_map->points[pt_search_idx_batch[j]].x;
                        matA0(j, 1) = surf_global_oth_frame_map->points[pt_search_idx_batch[j]].y;
                        matA0(j, 2) = surf_global_oth_frame_map->points[pt_search_idx_batch[j]].z;
                        matA0_local(j, 0) = surf_frames[search_idx]->points[pt_search_idx_batch[j]].x;
                        matA0_local(j, 1) = surf_frames[search_idx]->points[pt_search_idx_batch[j]].y;
                        matA0_local(j, 2) = surf_frames[search_idx]->points[pt_search_idx_batch[j]].z;
                        cent_x += matA0_local(j, 0);
                        cent_y += matA0_local(j, 1);
                        cent_z += matA0_local(j, 2);
                    }
                    normal_cent[3] = cent_x/5.;
                    normal_cent[4] = cent_y/5.;
                    normal_cent[5] = cent_z/5.;
                    // Get the norm of the plane using linear solver based on QR composition
                    Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                    double normInverse = 1 / norm.norm();
                    norm.normalize(); // get the unit norm
                    Eigen::Vector3d norm_local = matA0_local.colPivHouseholderQr().solve(matB0_local);
                    norm_local.normalize(); // get the unit norm
                    // Make sure that the plan is fit
                    bool planeValid = true;
                    for (int j = 0; j < 5; ++j) {
                        if (fabs(norm.x() * surf_global_oth_frame_map->points[pt_search_idx_batch[j]].x +
                                 norm.y() * surf_global_oth_frame_map->points[pt_search_idx_batch[j]].y +
                                 norm.z() * surf_global_oth_frame_map->points[pt_search_idx_batch[j]].z + normInverse) > 0.18) {
                            planeValid = false;
                            break;
                        }
                    }
                    // if one eigenvalue is significantly larger than the other two
                    if (planeValid) {
                        float pd = norm.x() * pt_in_gl_map_.x + norm.y() * pt_in_gl_map_.y + norm.z() *pt_in_gl_map_.z + normInverse;
                        float weight = 1 - 0.9 * fabs(pd) / sqrt(sqrt(pt_in_gl_map_.x * pt_in_gl_map_.x + pt_in_gl_map_.y * pt_in_gl_map_.y + pt_in_gl_map_.z * pt_in_gl_map_.z));
                        if(weight > 0.3) {
                            PointType normal;
                            normal.x = weight * norm.x();
                            normal.y = weight * norm.y();
                            normal.z = weight * norm.z();
                            normal.intensity = weight * normInverse;
                            normal_cent[0] = norm_local.x();
                            normal_cent[1] = norm_local.y();
                            normal_cent[2] = norm_local.z();
                            gl_vec_surf_cur_pts[idVec][search_idx]->points.push_back(pt_in_local_);
                            ++gl_vec_surf_res_cnt[idVec][search_idx];
                            gl_vec_surf_scores[idVec][search_idx].push_back(2.5*weight);
                            gl_vec_surf_normals_cents[idVec][search_idx].push_back(normal_cent);
                        }
                    }
                }
            }
        }
//        t_feature_association.tic_toc();
    }

    void featureSelection (int idx, Eigen::Quaterniond q, Eigen::Vector3d t) {
//        Timer t_feature_select("FeatureSelector");

        int idVec = idx-keyframe_idx[keyframe_idx.size()-slide_window_width] + 1;

        int surf_pts_size = vec_surf_cur_pts[idVec]->points.size();
        vec_surf_cur_pts[idVec]->resize(surf_pts_size);
        vec_surf_normal[idVec]->resize(surf_pts_size);
        if (surf_pts_size < 1) return;
        int org_rand_set_num = rand_set_num;
        int org_feature_res_num = feature_res_num;

        if (surf_pts_size - 1 < feature_res_num) {
            return;
            feature_res_num = surf_pts_size - 1;
            rand_set_num = surf_pts_size - 1;
        }
        if (surf_pts_size - 1 < rand_set_num) {
            rand_set_num = surf_pts_size - 1;
        }
        if (surf_pts_size - feature_res_num < rand_set_num) {
            rand_set_num = surf_pts_size - feature_res_num - 1;
        }
        /* init feature selection para */
        double sum_LogDeterminant = 0;
        Eigen::Matrix<double, 6, 6> JTJ_selected_feature_sum = Eigen::Matrix<double, 6, 6>::Zero();

        /* init temp variables */                       
        pcl::PointCloud<PointType>::Ptr surf_cur_pt_temp (new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surf_cur_normal_temp (new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surf_cur_pt_temp_less (new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surf_cur_normal_temp_less (new pcl::PointCloud<PointType>());
        vector<double> surf_score_temp;
        int surf_res_cnt_temp = 0;
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians_set;
        std::vector<double *> parameter_blocks_;

        /* init ceres problem parameter block */
        double Trans_[3];
        Trans_[0] = t[0];
        Trans_[1] = t[1];
        Trans_[2] = t[2];

        double Euler_[3];
        Eigen::Vector3d euler_2 = Eigen::Vector3d(0, 0, 0);
        toEulerAngle(q, euler_2);
        Euler_[0] = euler_2[0]; Euler_[1] = euler_2[1]; Euler_[2] = euler_2[2];

        /* random numbers generator */
        common::RandomGeneratorInt<int> rgi_;

        while (surf_cur_pt_temp->points.size() < feature_res_num && random_select) {
            int *rand_ids;
            rand_ids = rgi_.geneRandArrayNoRepeat(0, vec_surf_cur_pts[idVec]->points.size() - 1, rand_set_num);

            /* Search best feature points in random set*/
            double temp_res_LogDeterminant = -1;
            int selected_id = -1;

            for (int i=0; i<rand_set_num; i++) {
                int id = rand_ids[i];
                selected_id = id;
            }
            delete[]rand_ids;

            surf_cur_pt_temp->points.push_back(vec_surf_cur_pts[idVec]->points[selected_id]);
            surf_cur_normal_temp->points.push_back(vec_surf_normal[idVec]->points[selected_id]);
            surf_score_temp.push_back(vec_surf_scores[idVec][selected_id]);
            surf_res_cnt_temp++;

            pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
            inliers->indices.push_back(selected_id);

            pcl::ExtractIndices<PointType> extract1;
            extract1.setInputCloud(vec_surf_cur_pts[idVec]);
            extract1.setIndices(inliers);
            extract1.setNegative(true);
            extract1.filter(*vec_surf_cur_pts[idVec]);

            extract1.setInputCloud(vec_surf_normal[idVec]);
            extract1.setIndices(inliers);
            extract1.setNegative(true);
            extract1.filter(*vec_surf_normal[idVec]);

            vec_surf_scores[idVec].erase(vec_surf_scores[idVec].begin() + selected_id);
        }

        vec_surf_cur_pts[idVec]->clear();
        vec_surf_cur_pts[idVec] = surf_cur_pt_temp;
        vec_surf_normal[idVec]->clear();
        vec_surf_normal[idVec] = surf_cur_normal_temp;
        vec_surf_scores[idVec].clear();
        vec_surf_scores[idVec] = surf_score_temp;
        vec_surf_res_cnt[idVec] = surf_res_cnt_temp;

        feature_res_num = org_feature_res_num;
        rand_set_num = org_rand_set_num;
//        t_feature_select.tic_toc();
    }

    void globalFeatureSelection_Batch (int idx, int search_idx_start) {
//        Timer t_feature_select("globalFeatureSelection_Batch");

        int idVec = idx;

        for (int search_idx = search_idx_start; search_idx <= search_idx_start + 2*search_range; search_idx++) {
            if (search_idx == idx) continue;
            pcl::PointCloud<PointType>::Ptr surf_cur_pt_temp (new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surf_cur_normal_temp (new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surf_cur_pt_temp_less (new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surf_cur_normal_temp_less (new pcl::PointCloud<PointType>());
            vector<double> surf_score_temp;
            int surf_res_cnt_temp = 0;
            vector<vector<double>> surf_normals_cents_temp;

            /* random numbers generator */
            common::RandomGeneratorInt<int> rgi_;

            int org_batch_feature_res_num = batch_feature_res_num;
            int org_batch_rand_set_num = batch_rand_set_num;

            if (gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.size() - 1 < batch_feature_res_num ||
                    gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.size() < 50) {
                return;
            }
            if (gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.size() - 1 < batch_rand_set_num) {
                batch_rand_set_num = gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.size() - 1;
            }
            if (gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.size() - batch_feature_res_num < batch_rand_set_num) {
                batch_rand_set_num = gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.size() - batch_feature_res_num - 1;
            }
            while (surf_cur_pt_temp->points.size() < batch_feature_res_num) {
                int *rand_ids;
                rand_ids = rgi_.geneRandArrayNoRepeat(0, gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.size() - 1, batch_rand_set_num);

                /* Search best feature points in random set*/
                int selected_id = -1;

                for (int i=0; i<batch_feature_res_num; i++) {
                    int id = rand_ids[i];
                    selected_id = id;
                    surf_cur_pt_temp->points.push_back(gl_vec_surf_cur_pts_startend[idVec][search_idx]->points[selected_id]);
                    surf_score_temp.push_back(gl_vec_surf_scores_startend[idVec][search_idx][selected_id]);
                    surf_res_cnt_temp++;
                    surf_normals_cents_temp.push_back(gl_vec_surf_normals_cents_startend[idVec][search_idx][selected_id]);
                }
                gl_vec_surf_cur_pts_startend[idVec][search_idx]->clear();
                gl_vec_surf_cur_pts_startend[idVec][search_idx] = surf_cur_pt_temp;
                gl_vec_surf_scores_startend[idVec][search_idx].clear();
                gl_vec_surf_scores_startend[idVec][search_idx] = surf_score_temp;
                gl_vec_surf_res_cnt_startend[idVec][search_idx] = surf_res_cnt_temp;
                gl_vec_surf_normals_cents_startend[idVec][search_idx].clear();
                gl_vec_surf_normals_cents_startend[idVec][search_idx] = surf_normals_cents_temp;
                delete[] rand_ids;
            }
            batch_feature_res_num = org_batch_feature_res_num;
            batch_rand_set_num = org_batch_rand_set_num;
        }

//        t_feature_select.tic_toc();

    }

    void globalFeatureSelectionAdd_Batch (int idx, int search_idx_start) {
//        Timer t_feature_select("globalFeatureSelectionAdd_Batch");

        int idVec = idx;

        for (int search_idx = search_idx_start; search_idx <= search_idx_start + 2*search_range; search_idx++) {
            if (search_idx == idx) continue;
            pcl::PointCloud<PointType>::Ptr surf_cur_pt_temp (new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surf_cur_normal_temp (new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surf_cur_pt_temp_less (new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surf_cur_normal_temp_less (new pcl::PointCloud<PointType>());
            vector<double> surf_score_temp;
            int surf_res_cnt_temp = 0;
            vector<vector<double>> surf_normals_cents_temp;

            /* random numbers generator */
            common::RandomGeneratorInt<int> rgi_;

            int org_batch_feature_res_num = batch_feature_res_num;
            int org_batch_rand_set_num = batch_rand_set_num;

            if (gl_vec_surf_cur_pts[idVec][search_idx]->points.size() <= batch_feature_res_num) {
                continue;
            }
            while (surf_cur_pt_temp->points.size() < batch_feature_res_num) {
                int *rand_ids;
                if (gl_vec_surf_cur_pts[idVec][search_idx]->points.size() - 1 <= batch_feature_res_num)
                    cout << "rand set larger than org set " << endl;
                rand_ids = rgi_.geneRandArrayNoRepeat(0, gl_vec_surf_cur_pts[idVec][search_idx]->points.size() - 1, batch_feature_res_num);


                /* Search best feature points in random set*/
                int selected_id = -1;

                for (int i=0; i<batch_feature_res_num; i++) {
                    int id = rand_ids[i];
                    selected_id = id;
                    surf_cur_pt_temp->points.push_back(gl_vec_surf_cur_pts[idVec][search_idx]->points[selected_id]);
                    surf_score_temp.push_back(gl_vec_surf_scores[idVec][search_idx][selected_id]);
                    surf_res_cnt_temp++;
                    surf_normals_cents_temp.push_back(gl_vec_surf_normals_cents[idVec][search_idx][selected_id]);
                }

                gl_vec_surf_cur_pts[idVec][search_idx]->clear();
                gl_vec_surf_cur_pts[idVec][search_idx] = surf_cur_pt_temp;
                gl_vec_surf_scores[idVec][search_idx].clear();
                gl_vec_surf_scores[idVec][search_idx] = surf_score_temp;
                gl_vec_surf_res_cnt[idVec][search_idx] = surf_res_cnt_temp;
                gl_vec_surf_normals_cents[idVec][search_idx].clear();
                gl_vec_surf_normals_cents[idVec][search_idx] = surf_normals_cents_temp;

                delete[] rand_ids;
            }
            batch_feature_res_num = org_batch_feature_res_num;
            batch_rand_set_num = org_batch_rand_set_num;
        }

//        t_feature_select.tic_toc();

    }

    static void toEulerAngle(const Quaterniond& q, Eigen::Vector3d &euler)
    {
        // roll (x-axis rotation)
        double sinr_cosp = +2.0 * (q.w() * q.x() + q.y() * q.z());
        double cosr_cosp = +1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
        euler[0] = atan2(sinr_cosp, cosr_cosp);

        // pitch (y-axis rotation)
        double sinp = +2.0 * (q.w() * q.y() - q.z() * q.x());
        if (fabs(sinp) >= 1)
        euler[1] = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
        else
        euler[1] = asin(sinp);

        // yaw (z-axis rotation)
        double siny_cosp = +2.0 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
        euler[2] = atan2(siny_cosp, cosy_cosp);
    }

    Eigen::Vector3d Translation_GNSS_IMU(double* &tmpQuat_){
        Eigen::Matrix3d R_tmp = Eigen::Quaterniond(tmpQuat_[0], tmpQuat_[1], tmpQuat_[2], tmpQuat_[3]).toRotationMatrix();
        Eigen::Vector3d t_gnss_imu = T_gnss_imu.block<3, 3>(0, 0) * R_tmp * T_gnss_imu.block<3, 1>(0, 3);
        return t_gnss_imu;
    }

    /* optimize the keyframes */
    void saveKeyFramesAndFactors() {
//        Timer t_sFF("saveKeyFramesAndFactors");
        abs_poses.push_back(abs_pose);
        keyframe_id_in_frame.push_back(each_odom_buf.size()-1); // each_odom_buf max is 50, for example, 1, 5, 10, 15, ..

        pcl::PointCloud<PointType>::Ptr surfEachFrame(new pcl::PointCloud<PointType>());

        *surfEachFrame = *surf_last_ds;
        surf_frames.push_back(surfEachFrame);

        //record index of kayframe on imu preintegration poses
        keyframe_idx.push_back(abs_poses.size()-1);

        keyframe_time.push_back(odom_cur->header.stamp.toSec()); // 3Hz roughly

        double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;

        int i = idx_imu;
        Eigen::Quaterniond tmpOrient;
        double timeodom_cur = odom_cur->header.stamp.toSec();
        if(imu_buf[i]->header.stamp.toSec() > timeodom_cur)
            ROS_WARN("Timestamp not synchronized, please check your hardware!");
        while(imu_buf[i]->header.stamp.toSec() < timeodom_cur) {
            double t = imu_buf[i]->header.stamp.toSec();
            if (cur_time_imu < 0)
                cur_time_imu = t;
            double dt = t - cur_time_imu;
            cur_time_imu = imu_buf[i]->header.stamp.toSec();
            dx = imu_buf[i]->linear_acceleration.x;
            dy = imu_buf[i]->linear_acceleration.y;
            dz = imu_buf[i]->linear_acceleration.z;
            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            rx = imu_buf[i]->angular_velocity.x;
            ry = imu_buf[i]->angular_velocity.y;
            rz = imu_buf[i]->angular_velocity.z;

            tmpOrient = Eigen::Quaterniond(imu_buf[i]->orientation.w,
                                           imu_buf[i]->orientation.x,
                                           imu_buf[i]->orientation.y,
                                           imu_buf[i]->orientation.z);
            processIMU(dt, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
            i++;
            if(i >= imu_buf.size())
                break;
        }
        imu_idx_in_kf.push_back(i - 1);

        if(i < imu_buf.size()) {
            double dt1 = timeodom_cur - cur_time_imu;
            double dt2 = imu_buf[i]->header.stamp.toSec() - timeodom_cur;

            double w1 = dt2 / (dt1 + dt2);
            double w2 = dt1 / (dt1 + dt2);

            Eigen::Quaterniond orient1 = Eigen::Quaterniond(imu_buf[i]->orientation.w,
                                                            imu_buf[i]->orientation.x,
                                                            imu_buf[i]->orientation.y,
                                                            imu_buf[i]->orientation.z);
            tmpOrient = tmpOrient.slerp(w2, orient1);

            dx = w1 * dx + w2 * imu_buf[i]->linear_acceleration.x;
            dy = w1 * dy + w2 * imu_buf[i]->linear_acceleration.y;
            dz = w1 * dz + w2 * imu_buf[i]->linear_acceleration.z;

            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            rx = w1 * rx + w2 * imu_buf[i]->angular_velocity.x;
            ry = w1 * ry + w2 * imu_buf[i]->angular_velocity.y;
            rz = w1 * rz + w2 * imu_buf[i]->angular_velocity.z;
            processIMU(dt1, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
        }
        cur_time_imu = timeodom_cur;
        vector<double> tmpSpeedBias;
        tmpSpeedBias.push_back(Vs.back().x());
        tmpSpeedBias.push_back(Vs.back().y());
        tmpSpeedBias.push_back(Vs.back().z());
        tmpSpeedBias.push_back(Bas.back().x());
        tmpSpeedBias.push_back(Bas.back().y());
        tmpSpeedBias.push_back(Bas.back().z());
        tmpSpeedBias.push_back(Bgs.back().x());
        tmpSpeedBias.push_back(Bgs.back().y());
        tmpSpeedBias.push_back(Bgs.back().z());
        para_speed_bias.push_back(tmpSpeedBias);
        vector<double> tmp_rcv_dt = {0, 0, 0};
        rcv_dt.push_back(tmp_rcv_dt);
        idx_imu = i;

        PointXYZI latestPose;
        PointPoseInfo latestPoseInfo;
        latestPose.x = Ps.back().x();
        latestPose.y = Ps.back().y();
        latestPose.z = Ps.back().z();
        latestPose.intensity = pose_keyframe->points.size();
        pose_keyframe->push_back(latestPose);

        latestPoseInfo.x = Ps.back().x();
        latestPoseInfo.y = Ps.back().y();
        latestPoseInfo.z = Ps.back().z();
        Eigen::Quaterniond qs_last(Rs.back());
        latestPoseInfo.qw = qs_last.w();
        latestPoseInfo.qx = qs_last.x();
        latestPoseInfo.qy = qs_last.y();
        latestPoseInfo.qz = qs_last.z();
        latestPoseInfo.idx = pose_keyframe->points.size();
        latestPoseInfo.time = time_new_odom;

        pose_info_keyframe->push_back(latestPoseInfo);

        //optimize sliding window
        num_kf_sliding++;
        if(num_kf_sliding >= 1 || !first_opt) {
            optimizeSlidingWindowWithLandMark();
            num_kf_sliding = 0;
        }

        /* optimize local factor graph */
        if (pose_keyframe->points.size() == slide_window_width) {
            pose_each_frame->push_back(pose_keyframe->points[0]);
            pose_info_each_frame->push_back(pose_info_keyframe->points[0]);
        }
        else if(pose_keyframe->points.size() > slide_window_width) {
            int ii = imu_idx_in_kf[imu_idx_in_kf.size() - slide_window_width - 1];
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            Eigen::Vector3d Ptmp = Ps[Ps.size() - slide_window_width];
            Eigen::Vector3d Vtmp = Vs[Ps.size() - slide_window_width];
            Eigen::Matrix3d Rtmp = Rs[Ps.size() - slide_window_width];
            Eigen::Vector3d Batmp = Eigen::Vector3d::Zero();
            Eigen::Vector3d Bgtmp = Eigen::Vector3d::Zero();

            for(int i = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1] + 1;
                i < keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width]; i++) {

                double dt1 = each_odom_buf[i-1]->header.stamp.toSec() - imu_buf[ii]->header.stamp.toSec();
                double dt2 = imu_buf[ii+1]->header.stamp.toSec() - each_odom_buf[i-1]->header.stamp.toSec();

                double w1 = dt2 / (dt1 + dt2);
                double w2 = dt1 / (dt1 + dt2);
                dx = w1 * imu_buf[ii]->linear_acceleration.x + w2 * imu_buf[ii+1]->linear_acceleration.x;
                dy = w1 * imu_buf[ii]->linear_acceleration.y + w2 * imu_buf[ii+1]->linear_acceleration.y;
                dz = w1 * imu_buf[ii]->linear_acceleration.z + w2 * imu_buf[ii+1]->linear_acceleration.z;

                rx = w1 * imu_buf[ii]->angular_velocity.x + w2 * imu_buf[ii+1]->angular_velocity.x;
                ry = w1 * imu_buf[ii]->angular_velocity.y + w2 * imu_buf[ii+1]->angular_velocity.y;
                rz = w1 * imu_buf[ii]->angular_velocity.z + w2 * imu_buf[ii+1]->angular_velocity.z;
                Eigen::Vector3d a0(dx, dy, dz);
                Eigen::Vector3d gy0(rx, ry, rz);
                ii++;
                double integStartTime = each_odom_buf[i-1]->header.stamp.toSec();

                while(imu_buf[ii]->header.stamp.toSec() < each_odom_buf[i]->header.stamp.toSec()) {
                    double t = imu_buf[ii]->header.stamp.toSec();
                    double dt = t - integStartTime;
                    integStartTime = imu_buf[ii]->header.stamp.toSec();
                    dx = imu_buf[ii]->linear_acceleration.x;
                    dy = imu_buf[ii]->linear_acceleration.y;
                    dz = imu_buf[ii]->linear_acceleration.z;

                    rx = imu_buf[ii]->angular_velocity.x;
                    ry = imu_buf[ii]->angular_velocity.y;
                    rz = imu_buf[ii]->angular_velocity.z;

                    if(dx > 15.0) dx = 15.0;
                    if(dy > 15.0) dy = 15.0;
                    if(dz > 18.0) dz = 18.0;

                    if(dx < -15.0) dx = -15.0;
                    if(dy < -15.0) dy = -15.0;
                    if(dz < -18.0) dz = -18.0;

                    Eigen::Vector3d a1(dx, dy, dz);
                    Eigen::Vector3d gy1(rx, ry, rz);

                    Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
                    Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
                    Rtmp *= deltaQ(un_gyr * dt).toRotationMatrix();
                    Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
                    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
                    Ptmp += dt * Vtmp + 0.5 * dt * dt * un_acc;
                    Vtmp += dt * un_acc;

                    a0 = a1;
                    gy0 = gy1;

                    ii++;
                }

                dt1 = each_odom_buf[i]->header.stamp.toSec() - imu_buf[ii-1]->header.stamp.toSec();
                dt2 = imu_buf[ii]->header.stamp.toSec() - each_odom_buf[i]->header.stamp.toSec();
                w1 = dt2 / (dt1 + dt2);
                w2 = dt1 / (dt1 + dt2);
                dx = w1 * dx + w2 * imu_buf[ii]->linear_acceleration.x;
                dy = w1 * dy + w2 * imu_buf[ii]->linear_acceleration.y;
                dz = w1 * dz + w2 * imu_buf[ii]->linear_acceleration.z;

                rx = w1 * rx + w2 * imu_buf[ii]->angular_velocity.x;
                ry = w1 * ry + w2 * imu_buf[ii]->angular_velocity.y;
                rz = w1 * rz + w2 * imu_buf[ii]->angular_velocity.z;

                if(dx > 15.0) dx = 15.0;
                if(dy > 15.0) dy = 15.0;
                if(dz > 18.0) dz = 18.0;

                if(dx < -15.0) dx = -15.0;
                if(dy < -15.0) dy = -15.0;
                if(dz < -18.0) dz = -18.0;

                Eigen::Vector3d a1(dx, dy, dz);
                Eigen::Vector3d gy1(rx, ry, rz);

                Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
                Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
                Rtmp *= deltaQ(un_gyr * dt1).toRotationMatrix();
                Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
                Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
                Ptmp += dt1 * Vtmp + 0.5 * dt1 * dt1 * un_acc;
                Vtmp += dt1 * un_acc;

                ii--;

                Eigen::Quaterniond qqq(Rtmp);

                PointXYZI latestPose;
                PointPoseInfo latestPoseInfo;
                latestPose.x = Ptmp.x();
                latestPose.y = Ptmp.y();
                latestPose.z = Ptmp.z();
                pose_each_frame->push_back(latestPose);

                latestPoseInfo.x = Ptmp.x();
                latestPoseInfo.y = Ptmp.y();
                latestPoseInfo.z = Ptmp.z();
                latestPoseInfo.qw = qqq.w();
                latestPoseInfo.qx = qqq.x();
                latestPoseInfo.qy = qqq.y();
                latestPoseInfo.qz = qqq.z();
                latestPoseInfo.time = each_odom_buf[i]->header.stamp.toSec();
                pose_info_each_frame->push_back(latestPoseInfo);
            }

            pose_each_frame->push_back(pose_keyframe->points[pose_keyframe->points.size() - slide_window_width]);
            pose_info_each_frame->push_back(pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width]);
            int j = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width];

            double dt1 = each_odom_buf[j-1]->header.stamp.toSec() - imu_buf[ii]->header.stamp.toSec();
            double dt2 = imu_buf[ii+1]->header.stamp.toSec() - each_odom_buf[j-1]->header.stamp.toSec();
            double w1 = dt2 / (dt1 + dt2);
            double w2 = dt1 / (dt1 + dt2);
            dx = w1 * imu_buf[ii]->linear_acceleration.x + w2 * imu_buf[ii+1]->linear_acceleration.x;
            dy = w1 * imu_buf[ii]->linear_acceleration.y + w2 * imu_buf[ii+1]->linear_acceleration.y;
            dz = w1 * imu_buf[ii]->linear_acceleration.z + w2 * imu_buf[ii+1]->linear_acceleration.z;

            rx = w1 * imu_buf[ii]->angular_velocity.x + w2 * imu_buf[ii+1]->angular_velocity.x;
            ry = w1 * imu_buf[ii]->angular_velocity.y + w2 * imu_buf[ii+1]->angular_velocity.y;
            rz = w1 * imu_buf[ii]->angular_velocity.z + w2 * imu_buf[ii+1]->angular_velocity.z;

            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            Eigen::Vector3d a0(dx, dy, dz);
            Eigen::Vector3d gy0(rx, ry, rz);
            ii++;
            double integStartTime = each_odom_buf[j-1]->header.stamp.toSec();

            while(imu_buf[ii]->header.stamp.toSec() < each_odom_buf[j]->header.stamp.toSec()) {
                double t = imu_buf[ii]->header.stamp.toSec();
                double dt = t - integStartTime;
                integStartTime = imu_buf[ii]->header.stamp.toSec();
                dx = imu_buf[ii]->linear_acceleration.x;
                dy = imu_buf[ii]->linear_acceleration.y;
                dz = imu_buf[ii]->linear_acceleration.z;

                rx = imu_buf[ii]->angular_velocity.x;
                ry = imu_buf[ii]->angular_velocity.y;
                rz = imu_buf[ii]->angular_velocity.z;

                if(dx > 15.0) dx = 15.0;
                if(dy > 15.0) dy = 15.0;
                if(dz > 18.0) dz = 18.0;

                if(dx < -15.0) dx = -15.0;
                if(dy < -15.0) dy = -15.0;
                if(dz < -18.0) dz = -18.0;

                Eigen::Vector3d a1(dx, dy, dz);
                Eigen::Vector3d gy1(rx, ry, rz);

                Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
                Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
                Rtmp *= deltaQ(un_gyr * dt).toRotationMatrix();
                Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
                Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
                Ptmp += dt * Vtmp + 0.5 * dt * dt * un_acc;
                Vtmp += dt * un_acc;

                a0 = a1;
                gy0 = gy1;

                ii++;
            }

            dt1 = each_odom_buf[j]->header.stamp.toSec() - imu_buf[ii-1]->header.stamp.toSec();
            dt2 = imu_buf[ii]->header.stamp.toSec() - each_odom_buf[j]->header.stamp.toSec();
            w1 = dt2 / (dt1 + dt2);
            w2 = dt1 / (dt1 + dt2);
            dx = w1 * dx + w2 * imu_buf[ii]->linear_acceleration.x;
            dy = w1 * dy + w2 * imu_buf[ii]->linear_acceleration.y;
            dz = w1 * dz + w2 * imu_buf[ii]->linear_acceleration.z;

            rx = w1 * rx + w2 * imu_buf[ii]->angular_velocity.x;
            ry = w1 * ry + w2 * imu_buf[ii]->angular_velocity.y;
            rz = w1 * rz + w2 * imu_buf[ii]->angular_velocity.z;

            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            Eigen::Vector3d a1(dx, dy, dz);
            Eigen::Vector3d gy1(rx, ry, rz);

            Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
            Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
            Rtmp *= deltaQ(un_gyr * dt1).toRotationMatrix();
            Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
            Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
            Ptmp += dt1 * Vtmp + 0.5 * dt1 * dt1 * un_acc;
            Vtmp += dt1 * un_acc;

            vector<double*> paraBetweenEachFrame;
            int numPara = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width] - keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1];
            double dQuat[numPara][4];
            double dTrans[numPara][3];
            for(int i = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1] + 1;
                i < keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width]; i++) {
                Eigen::Vector3d tmpTrans = Eigen::Vector3d(pose_each_frame->points[i].x,
                                                           pose_each_frame->points[i].y,
                                                           pose_each_frame->points[i].z) -
                        Eigen::Vector3d(pose_each_frame->points[i-1].x,
                        pose_each_frame->points[i-1].y,
                        pose_each_frame->points[i-1].z);
                tmpTrans = Eigen::Quaterniond(pose_info_each_frame->points[i-1].qw,
                        pose_info_each_frame->points[i-1].qx,
                        pose_info_each_frame->points[i-1].qy,
                        pose_info_each_frame->points[i-1].qz).inverse() * tmpTrans;
                dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][0] = tmpTrans.x();
                dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][1] = tmpTrans.y();
                dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][2] = tmpTrans.z();
                paraBetweenEachFrame.push_back(dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1]);

                Eigen::Quaterniond tmpQuat = Eigen::Quaterniond(pose_info_each_frame->points[i-1].qw,
                        pose_info_each_frame->points[i-1].qx,
                        pose_info_each_frame->points[i-1].qy,
                        pose_info_each_frame->points[i-1].qz).inverse() *
                        Eigen::Quaterniond(pose_info_each_frame->points[i].qw,
                                           pose_info_each_frame->points[i].qx,
                                           pose_info_each_frame->points[i].qy,
                                           pose_info_each_frame->points[i].qz);
                dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][0] = tmpQuat.w();
                dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][1] = tmpQuat.x();
                dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][2] = tmpQuat.y();
                dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][3] = tmpQuat.z();
                paraBetweenEachFrame.push_back(dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1]);
//                Eigen::Vector3d euler;
//                toEulerAngle(Eigen::Quaterniond(tmpQuat.w(), tmpQuat.x(), tmpQuat.y(), tmpQuat.z()), euler);
            }
            int jj = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width];

            Eigen::Vector3d tmpTrans = Ptmp - Eigen::Vector3d(pose_each_frame->points[jj-1].x,
                    pose_each_frame->points[jj-1].y,
                    pose_each_frame->points[jj-1].z);
            tmpTrans = Eigen::Quaterniond(pose_info_each_frame->points[jj-1].qw,
                    pose_info_each_frame->points[jj-1].qx,
                    pose_info_each_frame->points[jj-1].qy,
                    pose_info_each_frame->points[jj-1].qz).inverse() * tmpTrans;

            dTrans[numPara-1][0] = tmpTrans.x();
            dTrans[numPara-1][1] = tmpTrans.y();
            dTrans[numPara-1][2] = tmpTrans.z();
            paraBetweenEachFrame.push_back(dTrans[numPara-1]);

            Eigen::Quaterniond qtmp(Rtmp);
            Eigen::Quaterniond tmpQuat = Eigen::Quaterniond(pose_info_each_frame->points[jj-1].qw,
                    pose_info_each_frame->points[jj-1].qx,
                    pose_info_each_frame->points[jj-1].qy,
                    pose_info_each_frame->points[jj-1].qz).inverse() * qtmp;
            dQuat[numPara-1][0] = tmpQuat.w();
            dQuat[numPara-1][1] = tmpQuat.x();
            dQuat[numPara-1][2] = tmpQuat.y();
            dQuat[numPara-1][3] = tmpQuat.z();
            paraBetweenEachFrame.push_back(dQuat[numPara-1]);

            optimizeLocalGraph(paraBetweenEachFrame);
        }

        // local lc FGO
        if (LCinLocal) {
            if (pose_keyframe->points.size() < slide_window_width) return;
            addLIOFactor();
            addGNSSFactor();
            isam->update(local_pose_graph, local_init_estimate);
            if (GNSSAdded)
            {
                isam->update();
                isam->update();
                isam->update();
                isam->update();
                isam->update();
            }
            GNSSAdded = false;
            local_pose_graph.resize(0);
            local_init_estimate.clear();

            isamCurrentEstimate = isam->calculateEstimate();
            poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);
            publishLCOdometry();
        }

        if (!loop_closure_on)
            return;

        //add poses to global graph
        if (pose_keyframe->points.size() == slide_window_width) {
            gtsam::Rot3 rotation = gtsam::Rot3::Quaternion(pose_info_each_frame->points[0].qw,
                                                           pose_info_each_frame->points[0].qx,
                                                           pose_info_each_frame->points[0].qy,
                                                           pose_info_each_frame->points[0].qz);
            gtsam::Point3 transition = gtsam::Point3(pose_each_frame->points[0].x,
                                                     pose_each_frame->points[0].y,
                                                     pose_each_frame->points[0].z);

            // Initialization for global pose graph
            global_pose_graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(rotation, transition), prior_noise));
            global_init_estimate.insert(0, gtsam::Pose3(rotation, transition));

            for (int i = 0; i < 7; ++i) {
                last_pose[i] = abs_poses[abs_poses.size()-slide_window_width][i];
            }
            select_pose.x = last_pose[4];
            select_pose.y = last_pose[5];
            select_pose.z = last_pose[6];
        }

            /* insert all the dense regular frames between two keyframes */
        else if(pose_keyframe->points.size() > slide_window_width) {
            for(int i = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1] + 1;
                i <= keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width]; i++) {

                gtsam::Rot3 rotationLast = gtsam::Rot3::Quaternion(pose_info_each_frame->points[i-1].qw,
                                                                   pose_info_each_frame->points[i-1].qx,
                                                                   pose_info_each_frame->points[i-1].qy,
                                                                   pose_info_each_frame->points[i-1].qz);
                gtsam::Point3 transitionLast = gtsam::Point3(pose_each_frame->points[i-1].x,
                                                             pose_each_frame->points[i-1].y,
                                                             pose_each_frame->points[i-1].z);

                gtsam::Rot3 rotationCur = gtsam::Rot3::Quaternion(pose_info_each_frame->points[i].qw,
                                                                  pose_info_each_frame->points[i].qx,
                                                                  pose_info_each_frame->points[i].qy,
                                                                  pose_info_each_frame->points[i].qz);
                gtsam::Point3 transitionCur = gtsam::Point3(pose_each_frame->points[i].x,
                                                            pose_each_frame->points[i].y,
                                                            pose_each_frame->points[i].z);
                gtsam::Pose3 poseFrom = gtsam::Pose3(rotationLast, transitionLast);
                gtsam::Pose3 poseTo = gtsam::Pose3(rotationCur, transitionCur);

                global_pose_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(i - 1,
                                                                         i,
                                                                         poseFrom.between(poseTo),
                                                                         odom_noise));
                global_init_estimate.insert(i, poseTo);
            }
        }

        isam->update(global_pose_graph, global_init_estimate);
        isam->update();

        global_pose_graph.resize(0);
        global_init_estimate.clear();

        if (pose_keyframe->points.size() > slide_window_width) {
            for (int i = 0; i < 7; ++i) {
                last_pose[i] = abs_poses[abs_poses.size()-slide_window_width][i];
            }
            select_pose.x = last_pose[4];
            select_pose.y = last_pose[5];
            select_pose.z = last_pose[6];
        }
//        t_sFF.tic_toc();

    }

    /* update the globally optimized pose estimation */
    void correctPoses() {
        if (loop_closed == true) {
            recent_surf_keyframes.clear();

            int numPoses = global_estimated.size();

            vector<Eigen::Quaterniond> quaternionRel;
            vector<Eigen::Vector3d> transitionRel;

            for(int i = abs_poses.size() - slide_window_width; i < abs_poses.size() - 1; i++) {
                Eigen::Quaterniond quaternionFrom(abs_poses[i][0],
                                                  abs_poses[i][1],
                                                  abs_poses[i][2],
                                                  abs_poses[i][3]);
                Eigen::Vector3d transitionFrom(abs_poses[i][4],
                                               abs_poses[i][5],
                                               abs_poses[i][6]);

                Eigen::Quaterniond quaternionTo(abs_poses[i+1][0],
                                                abs_poses[i+1][1],
                                                abs_poses[i+1][2],
                                                abs_poses[i+1][3]);
                Eigen::Vector3d transitionTo(abs_poses[i+1][4],
                                             abs_poses[i+1][5],
                                             abs_poses[i+1][6]);

                quaternionRel.push_back(quaternionFrom.inverse() * quaternionTo);
                transitionRel.push_back(quaternionFrom.inverse() * (transitionTo - transitionFrom));
            }

            for (int i = 0; i < numPoses; ++i) {
                pose_each_frame->points[i].x = global_estimated.at<gtsam::Pose3>(i).translation().x();
                pose_each_frame->points[i].y = global_estimated.at<gtsam::Pose3>(i).translation().y();
                pose_each_frame->points[i].z = global_estimated.at<gtsam::Pose3>(i).translation().z();

                pose_info_each_frame->points[i].x = pose_each_frame->points[i].x;
                pose_info_each_frame->points[i].y = pose_each_frame->points[i].y;
                pose_info_each_frame->points[i].z = pose_each_frame->points[i].z;
                pose_info_each_frame->points[i].qw = global_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().w();
                pose_info_each_frame->points[i].qx = global_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().x();
                pose_info_each_frame->points[i].qy = global_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().y();
                pose_info_each_frame->points[i].qz = global_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().z();
            }

            for(int i = 0; i <= pose_keyframe->points.size() - slide_window_width; i++) {
                pose_keyframe->points[i].x = pose_each_frame->points[keyframe_id_in_frame[i]].x;
                pose_keyframe->points[i].y = pose_each_frame->points[keyframe_id_in_frame[i]].y;
                pose_keyframe->points[i].z = pose_each_frame->points[keyframe_id_in_frame[i]].z;

                pose_info_keyframe->points[i].x = pose_each_frame->points[keyframe_id_in_frame[i]].x;
                pose_info_keyframe->points[i].y = pose_each_frame->points[keyframe_id_in_frame[i]].y;
                pose_info_keyframe->points[i].z = pose_each_frame->points[keyframe_id_in_frame[i]].z;
                pose_info_keyframe->points[i].qw = pose_info_each_frame->points[keyframe_id_in_frame[i]].qw;
                pose_info_keyframe->points[i].qx = pose_info_each_frame->points[keyframe_id_in_frame[i]].qx;
                pose_info_keyframe->points[i].qy = pose_info_each_frame->points[keyframe_id_in_frame[i]].qy;
                pose_info_keyframe->points[i].qz = pose_info_each_frame->points[keyframe_id_in_frame[i]].qz;

                abs_poses[i+1][0] = pose_info_keyframe->points[i].qw;
                abs_poses[i+1][1] = pose_info_keyframe->points[i].qx;
                abs_poses[i+1][2] = pose_info_keyframe->points[i].qy;
                abs_poses[i+1][3] = pose_info_keyframe->points[i].qz;
                abs_poses[i+1][4] = pose_info_keyframe->points[i].x;
                abs_poses[i+1][5] = pose_info_keyframe->points[i].y;
                abs_poses[i+1][6] = pose_info_keyframe->points[i].z;

                Rs[i+1] = Eigen::Quaterniond(abs_poses[i+1][0],
                                             abs_poses[i+1][1],
                                             abs_poses[i+1][2],
                                             abs_poses[i+1][3]).toRotationMatrix();

                Ps[i+1][0] = abs_poses[i+1][4];
                Ps[i+1][1] = abs_poses[i+1][5];
                Ps[i+1][2] = abs_poses[i+1][6];
            }

            for(int i = abs_poses.size() - slide_window_width; i < abs_poses.size() - 1; i++) {
                Eigen::Quaterniond integratedQuaternion(abs_poses[i][0],
                                                        abs_poses[i][1],
                                                        abs_poses[i][2],
                                                        abs_poses[i][3]);
                Eigen::Vector3d integratedTransition(abs_poses[i][4],
                                                     abs_poses[i][5],
                                                     abs_poses[i][6]);

                integratedTransition = integratedTransition + integratedQuaternion * transitionRel[i - abs_poses.size() + slide_window_width];
                integratedQuaternion = integratedQuaternion * quaternionRel[i - abs_poses.size() + slide_window_width];

                abs_poses[i+1][0] = integratedQuaternion.w();
                abs_poses[i+1][1] = integratedQuaternion.x();
                abs_poses[i+1][2] = integratedQuaternion.y();
                abs_poses[i+1][3] = integratedQuaternion.z();
                abs_poses[i+1][4] = integratedTransition.x();
                abs_poses[i+1][5] = integratedTransition.y();
                abs_poses[i+1][6] = integratedTransition.z();

                Rs[i+1] = Eigen::Quaterniond(abs_poses[i+1][0],
                                             abs_poses[i+1][1],
                                             abs_poses[i+1][2],
                                             abs_poses[i+1][3]).toRotationMatrix();

                Ps[i+1][0] = abs_poses[i+1][4];
                Ps[i+1][1] = abs_poses[i+1][5];
                Ps[i+1][2] = abs_poses[i+1][6];

                pose_keyframe->points[i].x = abs_poses[i+1][4];
                pose_keyframe->points[i].y = abs_poses[i+1][5];
                pose_keyframe->points[i].z = abs_poses[i+1][6];

                pose_info_keyframe->points[i].x = abs_poses[i+1][4];
                pose_info_keyframe->points[i].y = abs_poses[i+1][5];
                pose_info_keyframe->points[i].z = abs_poses[i+1][6];
                pose_info_keyframe->points[i].qw = abs_poses[i+1][0];
                pose_info_keyframe->points[i].qx = abs_poses[i+1][1];
                pose_info_keyframe->points[i].qy = abs_poses[i+1][2];
                pose_info_keyframe->points[i].qz = abs_poses[i+1][3];
            }

            abs_pose = abs_poses.back();
            for (int i = 0; i < 7; ++i) {
                last_pose[i] = abs_poses[abs_poses.size() - slide_window_width][i];
            }

            select_pose.x = last_pose[4];
            select_pose.y = last_pose[5];
            select_pose.z = last_pose[6];

            loop_closed = false;
            marg = false;
        }
    }

    void publishOdometry() {

        if(pose_info_keyframe->points.size() >= slide_window_width) {
            time_new_odom = keyframe_time[pose_info_keyframe->points.size()-slide_window_width];
            odom_mapping.header.stamp = ros::Time().fromSec(time_new_odom);
            odom_mapping.pose.pose.orientation.w = pose_info_keyframe->points[pose_info_keyframe->points.size()-slide_window_width].qw;

            odom_mapping.pose.pose.orientation.x = pose_info_keyframe->points[pose_info_keyframe->points.size()-slide_window_width].qx;

            odom_mapping.pose.pose.orientation.y = pose_info_keyframe->points[pose_info_keyframe->points.size()-slide_window_width].qy;
            odom_mapping.pose.pose.orientation.z = pose_info_keyframe->points[pose_info_keyframe->points.size()-slide_window_width].qz;
            odom_mapping.pose.pose.position.x = pose_info_keyframe->points[pose_info_keyframe->points.size()-slide_window_width].x;
            odom_mapping.pose.pose.position.y = pose_info_keyframe->points[pose_info_keyframe->points.size()-slide_window_width].y;
            odom_mapping.pose.pose.position.z = pose_info_keyframe->points[pose_info_keyframe->points.size()-slide_window_width].z;
            
            odom_mapping.twist.twist.linear.x = Bas.back().x(); // bias of imu acc
            odom_mapping.twist.twist.linear.y = Bas.back().y();
            odom_mapping.twist.twist.linear.z = Bas.back().z();
            odom_mapping.twist.twist.angular.x = Bgs.back().x();
            odom_mapping.twist.twist.angular.y = Bgs.back().y();
            odom_mapping.twist.twist.angular.z = Bgs.back().z();
            pub_odom.publish(odom_mapping);

            /* results for evo */
//            ofstream fout_evo(result_path_evo, ios::app); //GROUND_TRUTH_PATH_EVO
//            fout_evo.setf(ios::fixed, ios::floatfield);
//            fout_evo.precision(8);
//            fout_evo  << time_new_odom << ' ';
//            fout_evo.precision(8);
//            fout_evo  << odom_mapping.pose.pose.position.x << ' '
//                        << odom_mapping.pose.pose.position.y << ' '
//                        << odom_mapping.pose.pose.position.z << ' '
//                        << odom_mapping.pose.pose.orientation.x << ' '
//                        << odom_mapping.pose.pose.orientation.y << ' '
//                        << odom_mapping.pose.pose.orientation.z << ' '
//                        << odom_mapping.pose.pose.orientation.w << '\n';
//            fout_evo.close();

            geometry_msgs::PoseStamped enu_pose_msg;
            enu_pose_msg.header = rtk_ini_enu_path.header;

            /* GLIO pose estimation */
            enu_pos = Ps[Ps.size()-slide_window_width];
            enu_ypr = Utility::R2ypr(Rs[Rs.size()-1]);
            ecef_pos = anc_ecef + R_ecef_enu * enu_pos; // from enu to ecef
            Eigen::Vector3d lla_pos = ecef2geo(ecef_pos);

            /* publish the estimated pose in llh */
            sensor_msgs::NavSatFix gnss_lla_msg;
            gnss_lla_msg.header.stamp = ros::Time().fromSec(time_new_odom);
            gnss_lla_msg.header.frame_id = "GLIO";
            gnss_lla_msg.latitude = lla_pos.x();
            gnss_lla_msg.longitude = lla_pos.y();
            gnss_lla_msg.altitude = lla_pos.z();
            pub_gnss_lla.publish(gnss_lla_msg);

            /* publish the pose in ENU from TC */
            Eigen::Quaterniond enu_ori(Rs[Rs.size()-1]);
            enu_pose_msg.pose.position.x = Ps[Ps.size()-1].x();
            enu_pose_msg.pose.position.y = Ps[Ps.size()-1].y();
            enu_pose_msg.pose.position.z = Ps[Ps.size()-1].z();
            enu_pose_msg.pose.orientation.x = enu_ori.x();
            enu_pose_msg.pose.orientation.y = enu_ori.y();
            enu_pose_msg.pose.orientation.z = enu_ori.z();
            enu_pose_msg.pose.orientation.w = enu_ori.w();
            tc_enu_path.header = enu_pose_msg.header;
            tc_enu_path.header.frame_id = frame_id;
            tc_enu_path.poses.push_back(enu_pose_msg);
            pub_tc_enu_path.publish(tc_enu_path);

            /* write GNSS result to file */
            ofstream tc_sw_output(tc_sw_result_path, ios::app);
            tc_sw_output.setf(ios::fixed, ios::floatfield);
            tc_sw_output.precision(8);
            const double gnss_ts = time_new_odom + timeshift_IMUtoGNSS;
            gtime_t gtime = sec2time(gnss_ts);
            uint32_t gps_week  = 0;
            double gps_sec = time2gpst(gtime, &gps_week);
            tc_sw_output << time_new_odom << ',';
            tc_sw_output.precision(8);
            tc_sw_output << gps_week << ',';
            tc_sw_output << gps_sec << ',';
            
            tc_sw_output << lla_pos.x() << ','
                        << lla_pos.y() << ','
                        << lla_pos.z() << ','
                        << enu_ypr.x() << ','
                        << enu_ypr.y() << ','
                        << enu_ypr.z() << ','
                        << enu_pos[0] << ','
                        << enu_pos[1] << ','
                        << enu_pos[2] << '\n';
            tc_sw_output.close();


            // publish local-imu body tf
             static tf::TransformBroadcaster br;
             tf::Transform transform_enu_world;
             tf::Quaternion tf_q;

             // publish world-map tf
             tf::Transform transform_map_world;
             transform_map_world.setOrigin(tf::Vector3(odom_mapping.pose.pose.position.x, odom_mapping.pose.pose.position.y, odom_mapping.pose.pose.position.z));
             tf_q.setW(odom_mapping.pose.pose.orientation.w);
             tf_q.setX(odom_mapping.pose.pose.orientation.x);
             tf_q.setY(odom_mapping.pose.pose.orientation.y);
             tf_q.setZ(odom_mapping.pose.pose.orientation.z);
             transform_map_world.setRotation(tf_q);
             br.sendTransform(tf::StampedTransform(transform_map_world, odom_mapping.header.stamp, "GLIO", "GLIO_dyna"));

            
        }

        sensor_msgs::PointCloud2 msgs;

        if (pub_poses.getNumSubscribers() && pose_info_keyframe->points.size() >= slide_window_width) {
            pcl::toROSMsg(*pose_each_frame, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_poses.publish(msgs);

        }


        PointPoseInfo Tbl;
        Tbl.qw = q_bl.w();
        Tbl.qx = q_bl.x();
        Tbl.qy = q_bl.y();
        Tbl.qz = q_bl.z();
        Tbl.x = t_bl.x();
        Tbl.y = t_bl.y();
        Tbl.z = t_bl.z();

        // publish the surf feature points in lidar_init frame

        if (pub_surf.getNumSubscribers()) {
            for (int i = 0; i < surf_last_ds->points.size(); ++i) {
                transformPoint(&surf_last_ds->points[i], &surf_last_ds->points[i], q_bl, t_bl);
                transformPoint(&surf_last_ds->points[i], &surf_last_ds->points[i]);
            }
            pcl::PointCloud<PointType>::Ptr surf_res(new pcl::PointCloud<PointType>());
            Eigen::Matrix4f tf_initial = Eigen::Matrix4f::Identity();
            tf2::Quaternion gt_init_q;
            gt_init_q.setRPY(0 * 3.1415926/180, 0 * 3.1415926/180, 0 * 3.1415926/180);
            gt_init_q.normalize();
            tf_initial.block(0,0,3,3) = Eigen::Quaternionf(gt_init_q[3],gt_init_q[0],gt_init_q[1],gt_init_q[2]).toRotationMatrix();
            pcl::transformPointCloud(*surf_last_ds, *surf_res, tf_initial);
            pcl::toROSMsg(*surf_res, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_surf.publish(msgs);
        }

        if (pub_full.getNumSubscribers()) {
            for (int i = 0; i < full_cloud->points.size(); ++i) {
                transformPoint(&full_cloud->points[i], &full_cloud->points[i], q_bl, t_bl);
                transformPoint(&full_cloud->points[i], &full_cloud->points[i]);
            }
            pcl::toROSMsg(*full_cloud, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_full.publish(msgs);
        }

    }

    void publishLCOdometry() {

        std::ofstream lc_output(lc_result_path, std::ios::out);
        lc_output.close();
//        std::ofstream res_lc_evo_output(lc_result_path_evo, std::ios::out);
//        res_lc_evo_output.close();

        nav_msgs::Path lc_enu_path;
        lc_enu_path.header.frame_id = frame_id;
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses - 1; ++i)
        {
            double time_frame = pose_info_keyframe->points[i].time;
            geometry_msgs::PoseStamped lc_pose;
            lc_pose.header.stamp = ros::Time().fromSec(time_frame);
            tf::Quaternion q_tmp = tf::createQuaternionFromRPY(isamCurrentEstimate.at<Pose3>(i).rotation().roll(),
                                                               isamCurrentEstimate.at<Pose3>(i).rotation().pitch(),
                                                               isamCurrentEstimate.at<Pose3>(i).rotation().yaw());
            lc_pose.pose.orientation.w = q_tmp.w();
            lc_pose.pose.orientation.x = q_tmp.x();
            lc_pose.pose.orientation.y = q_tmp.y();
            lc_pose.pose.orientation.z = q_tmp.z();

            Eigen::Vector3d i_pos (isamCurrentEstimate.at<Pose3>(i).translation().x(),
                                   isamCurrentEstimate.at<Pose3>(i).translation().y(),
                                   isamCurrentEstimate.at<Pose3>(i).translation().z());

            lc_pose.pose.position.x = i_pos[0];
            lc_pose.pose.position.y = i_pos[1];
            lc_pose.pose.position.z = i_pos[2];

            lc_enu_path.poses.push_back(lc_pose);

            Eigen::Vector3d tmp_pos (lc_pose.pose.position.x,
                                     lc_pose.pose.position.y,
                                     lc_pose.pose.position.z);
            Eigen::Matrix3d tmp_rot = Eigen::Quaterniond (q_tmp.w(), q_tmp.x(), q_tmp.y(), q_tmp.z()).toRotationMatrix();

            /* results for evo */
//            ofstream fout_evo(lc_result_path_evo, ios::app); //
//            fout_evo.setf(ios::fixed, ios::floatfield);
//            fout_evo.precision(8);
//            fout_evo  << time_frame << ' ';
//            fout_evo.precision(8);
//            fout_evo  << lc_pose.pose.position.x << ' '
//                      << lc_pose.pose.position.y << ' '
//                      << lc_pose.pose.position.z << ' '
//                      << lc_pose.pose.orientation.x << ' '
//                      << lc_pose.pose.orientation.y << ' '
//                      << lc_pose.pose.orientation.z << ' '
//                      << lc_pose.pose.orientation.w << '\n';
//            fout_evo.close();

            geometry_msgs::PoseStamped lc_enu_pose_msg;
            lc_enu_pose_msg.header = rtk_ini_enu_path.header;

            /* publish the estimated pose in llh */
            enu_pos = tmp_pos;
            enu_ypr = Utility::R2ypr(tmp_rot);
            ecef_pos = anc_ecef + R_ecef_enu * enu_pos; // from enu to ecef
            Eigen::Vector3d lla_pos = ecef2geo(ecef_pos);

            sensor_msgs::NavSatFix gnss_lla_msg;
            gnss_lla_msg.header.stamp = ros::Time().fromSec(time_frame);
            gnss_lla_msg.header.frame_id = "GLIO";
            gnss_lla_msg.latitude = lla_pos.x();
            gnss_lla_msg.longitude = lla_pos.y();
            gnss_lla_msg.altitude = lla_pos.z();

            /* write GNSS result to file */
            ofstream lc_output(lc_result_path, ios::app);
            lc_output.setf(ios::fixed, ios::floatfield);
            lc_output.precision(8);
            gtime_t gtime = sec2time(time_frame);
            uint32_t gps_week  = 0;
            double gps_sec = time2gpst(gtime, &gps_week);
            lc_output << time_frame << ',';
            lc_output << gps_week << ',';
            lc_output << gps_sec << ',';

            lc_output << lla_pos.x() << ','
                        << lla_pos.y() << ','
                        << lla_pos.z() << ','
                        << enu_ypr.x() << ','
                        << enu_ypr.y() << ','
                        << enu_ypr.z() << ','
                        << enu_pos[0] << ','
                        << enu_pos[1] << ','
                        << enu_pos[2] << '\n';
            lc_output.close();
        }

        pub_lc_enu_path.publish(lc_enu_path);

    }

    void clearCloud() {
        surf_local_map->clear();
        surf_local_map_ds->clear();

        // if(surf_lasts_ds.size() > slide_window_width + 5) {
        //     surf_lasts_ds[surf_lasts_ds.size() - slide_window_width - 6]->clear();
        // }

        // if(pre_integrations.size() > slide_window_width + 5) {
        //     pre_integrations[pre_integrations.size() - slide_window_width - 6] = nullptr;
        // }

        // if(last_marginalization_parameter_blocks.size() > slide_window_width + 5) {
        //     last_marginalization_parameter_blocks[last_marginalization_parameter_blocks.size() - slide_window_width - 6] = nullptr;
        // }

        if(surf_lasts_ds.size() > 3 * slide_window_width + 1) {
            surf_lasts_ds[surf_lasts_ds.size() - 3 * slide_window_width - 2]->clear();
        }

//        if(surf_frames.size() > batch_fusion_width*1.5) {
//            surf_frames[surf_frames.size() - batch_fusion_width*1.5 - 1]->clear();
//        }

//        if(pre_integrations.size() > 2 * slide_window_width + 1) {
//            pre_integrations[pre_integrations.size() - 2 * slide_window_width - 2] = nullptr;
//        }

        /* window size of 10 */
//        if(last_marginalization_parameter_blocks.size() > 2 * slide_window_width + 1) {
//            last_marginalization_parameter_blocks[last_marginalization_parameter_blocks.size() - 2 * slide_window_width - 2] = nullptr;
//        }

        /* window size of 3 */
        // if(last_marginalization_parameter_blocks.size() > slide_window_width + 5) {
        //     last_marginalization_parameter_blocks[last_marginalization_parameter_blocks.size() - slide_window_width - 6] = nullptr;
        // }
    }

    void loopClosureThread() {
        if (!loop_closure_on)
            return;

        ros::Rate rate(1);
        while (ros::ok()) {
            rate.sleep();
            performLoopClosure();
        }
    }

    bool detectLoopClosure() {
        latest_key_frames->clear();
        latest_key_frames_ds->clear();
        his_key_frames->clear();
        his_key_frames_ds->clear();

        std::lock_guard<std::mutex> lock(mutual_exclusion);

        // Look for the closest key frames
        std::vector<int> pt_search_idxLoop;
        std::vector<float> pt_search_sq_distsLoop;

        kd_tree_his_key_poses->setInputCloud(pose_keyframe);
        kd_tree_his_key_poses->radiusSearch(select_pose, lc_search_radius, pt_search_idxLoop, pt_search_sq_distsLoop, 0);

        closest_his_idx = -1;
        for (int i = 0; i < pt_search_idxLoop.size(); ++i) {
            int idx = pt_search_idxLoop[i];
            if (abs(pose_info_keyframe->points[idx].time - time_new_odom) > lc_time_thres) {
                closest_his_idx = idx;
                break;
            }
        }

        if (closest_his_idx == -1)
            return false;
        else if(abs(time_last_loop - time_new_odom) < 0.2)
            return false;

//        ROS_INFO("******************* Loop closure ready to detect! *******************");

        // Combine the corner and surf frames to form the latest frame
        latest_frame_idx_loop = pose_keyframe->points.size() - slide_window_width;

        for (int j = 0; j < 6; ++j) {
            if (latest_frame_idx_loop-j < 0)
                continue;
            Eigen::Quaterniond q_po(pose_info_keyframe->points[latest_frame_idx_loop-j].qw,
                                    pose_info_keyframe->points[latest_frame_idx_loop-j].qx,
                                    pose_info_keyframe->points[latest_frame_idx_loop-j].qy,
                                    pose_info_keyframe->points[latest_frame_idx_loop-j].qz);

            Eigen::Vector3d t_po(pose_info_keyframe->points[latest_frame_idx_loop-j].x,
                                 pose_info_keyframe->points[latest_frame_idx_loop-j].y,
                                 pose_info_keyframe->points[latest_frame_idx_loop-j].z);

            Eigen::Quaterniond q_tmp = q_po * q_bl;
            Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

            *latest_key_frames += *transformCloud(surf_frames[latest_frame_idx_loop-j], q_tmp, t_tmp);
        }

        ds_filter_his_frames.setInputCloud(latest_key_frames);
        ds_filter_his_frames.filter(*latest_key_frames_ds);


        // Form the history frame for loop closure detection
        for (int j = -lc_map_width; j <= lc_map_width; ++j) {
            if (closest_his_idx + j < 0 || closest_his_idx + j > latest_frame_idx_loop)
                continue;

            Eigen::Quaterniond q_po(pose_info_keyframe->points[closest_his_idx+j].qw,
                                    pose_info_keyframe->points[closest_his_idx+j].qx,
                                    pose_info_keyframe->points[closest_his_idx+j].qy,
                                    pose_info_keyframe->points[closest_his_idx+j].qz);

            Eigen::Vector3d t_po(pose_info_keyframe->points[closest_his_idx+j].x,
                                 pose_info_keyframe->points[closest_his_idx+j].y,
                                 pose_info_keyframe->points[closest_his_idx+j].z);

            Eigen::Quaterniond q_tmp = q_po * q_bl;
            Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

            *his_key_frames += *transformCloud(surf_frames[closest_his_idx+j], q_tmp, t_tmp);
        }

        ds_filter_his_frames.setInputCloud(his_key_frames);
        ds_filter_his_frames.filter(*his_key_frames_ds);

        return true;
    }

    void performLoopClosure() {
        if (pose_keyframe->points.empty())
            return;

        if (!loop_to_close) {
            if (detectLoopClosure())
                loop_to_close = true;
            if (!loop_to_close)
                return;
        }

        loop_to_close = false;

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(30);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(5);

        icp.setInputSource(latest_key_frames_ds);
        icp.setInputTarget(his_key_frames_ds);
        pcl::PointCloud<PointType>::Ptr alignedCloud(new pcl::PointCloud<PointType>());
        icp.align(*alignedCloud);

        //std::cout << "ICP converg flag:" << icp.hasConverged() << ". Fitness score: " << icp.getFitnessScore() << endl;

        if (!icp.hasConverged() || icp.getFitnessScore() > lc_icp_thres)
            return;

        Timer t_loop("Loop Closure");
//        ROS_INFO("******************* Loop closure detected! *******************");

        Eigen::Matrix4d correctedTranform;
        correctedTranform = icp.getFinalTransformation().cast<double>();
        Eigen::Quaterniond quaternionIncre(correctedTranform.block<3, 3>(0, 0));
        Eigen::Vector3d transitionIncre(correctedTranform.block<3, 1>(0, 3));
        Eigen::Quaterniond quaternionToCorrect(pose_info_keyframe->points[latest_frame_idx_loop].qw,
                                               pose_info_keyframe->points[latest_frame_idx_loop].qx,
                                               pose_info_keyframe->points[latest_frame_idx_loop].qy,
                                               pose_info_keyframe->points[latest_frame_idx_loop].qz);
        Eigen::Vector3d transitionToCorrect(pose_info_keyframe->points[latest_frame_idx_loop].x,
                                            pose_info_keyframe->points[latest_frame_idx_loop].y,
                                            pose_info_keyframe->points[latest_frame_idx_loop].z);

        Eigen::Quaterniond quaternionCorrected = quaternionIncre * quaternionToCorrect;
        Eigen::Vector3d transitionCorrected = quaternionIncre * transitionToCorrect + transitionIncre;

        gtsam::Rot3 rotationFrom = gtsam::Rot3::Quaternion(quaternionCorrected.w(), quaternionCorrected.x(), quaternionCorrected.y(), quaternionCorrected.z());
        gtsam::Point3 transitionFrom = gtsam::Point3(transitionCorrected.x(), transitionCorrected.y(), transitionCorrected.z());

        gtsam::Rot3 rotationTo = gtsam::Rot3::Quaternion(pose_info_keyframe->points[closest_his_idx].qw,
                                                         pose_info_keyframe->points[closest_his_idx].qx,
                                                         pose_info_keyframe->points[closest_his_idx].qy,
                                                         pose_info_keyframe->points[closest_his_idx].qz);
        gtsam::Point3 transitionTo = gtsam::Point3(pose_info_keyframe->points[closest_his_idx].x,
                                                   pose_info_keyframe->points[closest_his_idx].y,
                                                   pose_info_keyframe->points[closest_his_idx].z);

        gtsam::Pose3 poseFrom = gtsam::Pose3(rotationFrom, transitionFrom);
        gtsam::Pose3 poseTo = gtsam::Pose3(rotationTo, transitionTo);
        gtsam::Vector vector6(6);
        double noiseScore = icp.getFitnessScore();
        vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        constraint_noise = gtsam::noiseModel::Diagonal::Variances(vector6);

        std::lock_guard<std::mutex> lock(mutual_exclusion);

        global_pose_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(keyframe_id_in_frame[latest_frame_idx_loop],
                                                                 keyframe_id_in_frame[closest_his_idx],
                                                                 poseFrom.between(poseTo),
                                                                 constraint_noise));
        isam->update(global_pose_graph);
        isam->update();
        global_pose_graph.resize(0);

        loop_closed = true;

        global_estimated = isam->calculateEstimate();
        correctPoses();

        if (last_marginalization_info) {
            delete last_marginalization_info;
        }
        last_marginalization_info = nullptr;

        time_last_loop = pose_info_keyframe->points[latest_frame_idx_loop].time;

//        ROS_INFO("******************* Loop closure finished! *******************");
        //t_loop.tic_toc();
    }

    void publishCompleteMap() {
        if (pose_keyframe->points.size() > 10) {
            for (int i = 0; i < pose_info_keyframe->points.size(); i = i + mapping_interval) {
                Eigen::Quaterniond q_po(pose_info_keyframe->points[i].qw,
                                        pose_info_keyframe->points[i].qx,
                                        pose_info_keyframe->points[i].qy,
                                        pose_info_keyframe->points[i].qz);

                Eigen::Vector3d t_po(pose_info_keyframe->points[i].x,
                                     pose_info_keyframe->points[i].y,
                                     pose_info_keyframe->points[i].z);

                Eigen::Quaterniond q_tmp = q_po * q_bl;
                Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

                PointPoseInfo Ttmp;
                Ttmp.qw = q_tmp.w();
                Ttmp.qx = q_tmp.x();
                Ttmp.qy = q_tmp.y();
                Ttmp.qz = q_tmp.z();
                Ttmp.x = t_tmp.x();
                Ttmp.y = t_tmp.y();
                Ttmp.z = t_tmp.z();

                *global_map += *transformCloud(full_clouds_ds[i], &Ttmp);
            }

            ds_filter_global_map.setInputCloud(global_map);
            ds_filter_global_map.filter(*global_map_ds);

            sensor_msgs::PointCloud2 msgs;
            pcl::toROSMsg(*global_map_ds, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_map.publish(msgs);
            global_map->clear();
            global_map_ds->clear();
        }
    }

    void mapVisualizationThread() {
        ros::Rate rate(0.038);
        while (ros::ok()) {
            return;
            rate.sleep();
            ROS_INFO("Publishing the map");
            publishCompleteMap();
        }

        if(!save_pcd)
            return;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;

        PointPoseInfo Tbl;
        Tbl.qw = q_bl.w();
        Tbl.qx = q_bl.x();
        Tbl.qy = q_bl.y();
        Tbl.qz = q_bl.z();
        Tbl.x = t_bl.x();
        Tbl.y = t_bl.y();
        Tbl.z = t_bl.z();

        for (int i = 0; i < pose_info_keyframe->points.size(); i = i + mapping_interval) {
            *global_map += *transformCloud(transformCloud(surf_frames[i], &Tbl), &pose_info_keyframe->points[i]);
        }
        ds_filter_global_map.setInputCloud(global_map);
        ds_filter_global_map.filter(*global_map_ds);
        std::string pcd_path = result_path + "global_map.pcd";
        pcl::io::savePCDFileASCII(pcd_path, *global_map_ds);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
        global_map->clear();
        global_map_ds->clear();
    }

    void backendFusionThread() {
        if (!enable_batch_fusion) return;
        ros::Rate rate(10);
        while (ros::ok()) {
            rate.sleep();
            optimizeBatchWithLandMark();
        }
    }

    void run() {
        if (new_surf && new_odom && new_each_odom && new_full_cloud) {
            new_surf = false;
            new_odom = false;
            new_each_odom = false;
            new_full_cloud = false;

            std::lock_guard<std::mutex> lock(mutual_exclusion);

            Timer t_map("Estimator");
            buildLocalMapWithLandMark(); // build local map
            downSampleCloud(); // downsample all the clouds
            saveKeyFramesAndFactors(); // 3D LiDAR Aided GNSS-RTK fusion
            publishOdometry(); // publish the latest
            clearCloud();
//            t_map.tic_toc();
//            runtime += t_map.toc();
//            cout<<"Estimator average run time: "<<runtime / each_odom_buf.size()<<endl;
//            cout<<"each_odom_buf.size()      : "<<each_odom_buf.size() <<endl;
        }
    }

};

 

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    ros::init(argc, argv, "GLIO");

    ROS_INFO("\033[1;32m---->\033[0m GLIO Started.");

    /* define the estimator */
    Estimator Estimator_;

    /* loop closure detection thread */
    std::thread threadLoopClosure(&Estimator::loopClosureThread, &Estimator_);

    /* map visualization thread, typically the map visualization can be time consuming */
    std::thread threadMapVisualization(&Estimator::mapVisualizationThread, &Estimator_);

    /* backend fusion thread */
    std::thread threadBackendFusion(&Estimator::backendFusionThread, &Estimator_);

    ros::Rate rate(200);
    bool publish_gt=false;
    while (ros::ok()) {
        ros::spinOnce();
        Estimator_.run();

        rate.sleep();
    }

    threadLoopClosure.join();
    threadMapVisualization.join();
    threadBackendFusion.join();

    return 0;
}
