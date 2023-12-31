 /* Copyright (C) 2019, Intelligent Positioning and Navigation Lab, Hong Kong Polytechnic University
 * 
 * This file is part of GraphGNSSLib.
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Weisong Wen (weisong.wen@connect.polyu.hk)
 *******************************************************/
#include "ros/ros.h"
#include "std_msgs/String.h"

#include <sstream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

/* RTKLIB Library */
#include "../RTKLIB/src/rtklib.h"
#include <stdio.h>
#include <assert.h>


static double a1[]={
  1585184.171,
 -6716599.430,
  3915742.905,
  7627233.455,
  9565990.879,
989457273.200
};
static double Q1[]={
0.227134,   0.112202,   0.112202,   0.112202,   0.112202,   0.103473,
0.112202,   0.227134,   0.112202,   0.112202,   0.112202,   0.103473,
0.112202,   0.112202,   0.227134,   0.112202,   0.112202,   0.103473,
0.112202,   0.112202,   0.112202,   0.227134,   0.112202,   0.103473,
0.112202,   0.112202,   0.112202,   0.112202,   0.227134,   0.103473,
0.103473,   0.103473,   0.103473,   0.103473,   0.103473,   0.434339
};
static double F1[]={
  1585184.000000,  1585184.000000,
 -6716599.000000, -6716600.000000,
  3915743.000000,  3915743.000000,
  7627234.000000,  7627233.000000,
  9565991.000000,  9565991.000000,
989457273.000000,989457273.000000
};
static double s1[]={
        3.507984,        3.708456, 
};
static double a2[]={
-13324172.755747,
-10668894.713608,
 -7157225.010770,
 -6149367.974367,
 -7454133.571066,
 -5969200.494550,
  8336734.058423,
  6186974.084502,
-17549093.883655,
-13970158.922370
};
static double Q2[]={
        0.446320,        0.223160,        0.223160,        0.223160,        0.223160,        0.572775,        0.286388,        0.286388,        0.286388,        0.286388,
        0.223160,        0.446320,        0.223160,        0.223160,        0.223160,        0.286388,        0.572775,        0.286388,        0.286388,        0.286388,
        0.223160,        0.223160,        0.446320,        0.223160,        0.223160,        0.286388,        0.286388,        0.572775,        0.286388,        0.286388,
        0.223160,        0.223160,        0.223160,        0.446320,        0.223160,        0.286388,        0.286388,        0.286388,        0.572775,        0.286388,
        0.223160,        0.223160,        0.223160,        0.223160,        0.446320,        0.286388,        0.286388,        0.286388,        0.286388,        0.572775,
        0.572775,        0.286388,        0.286388,        0.286388,        0.286388,        0.735063,        0.367531,        0.367531,        0.367531,        0.367531,
        0.286388,        0.572775,        0.286388,        0.286388,        0.286388,        0.367531,        0.735063,        0.367531,        0.367531,        0.367531,
        0.286388,        0.286388,        0.572775,        0.286388,        0.286388,        0.367531,        0.367531,        0.735063,        0.367531,        0.367531,
        0.286388,        0.286388,        0.286388,        0.572775,        0.286388,        0.367531,        0.367531,        0.367531,        0.735063,        0.367531,
        0.286388,        0.286388,        0.286388,        0.286388,        0.572775,        0.367531,        0.367531,        0.367531,        0.367531,        0.735063 
};
static double F2[]={
-13324188.000000,-13324188.000000,
-10668901.000000,-10668908.000000,
 -7157236.000000, -7157236.000000,
 -6149379.000000, -6149379.000000,
 -7454143.000000, -7454143.000000,
 -5969220.000000, -5969220.000000,
  8336726.000000,  8336717.000000,
  6186960.000000,  6186960.000000,
-17549108.000000,-17549108.000000,
-13970171.000000,-13970171.000000 
};
static double s2[]={
     1506.435789,     1612.811795
};

void utest1(void)
{
    int i,j,n,m,info;
    double F[6*2],s[2];
    
    n=6; m=2;
    info=lambda(n,m,a1,Q1,F,s);
    assert(info==0);

    for (j=0;j<m;j++) {
        for (i=0;i<n;i++) {
            assert(fabs(F[i+j*n]-F1[j+i*m])<1E-4);
        }
        assert(fabs(s[j]-s1[j])<1E-4);
    }
    printf("%s utest1 : OK\n",__FILE__);
}
void utest2(void)
{
    int i,j,n,m,info;
    double F[10*2],s[2];
    
    n=10; m=2;
    info=lambda(n,m,a2,Q2,F,s);
    assert(info==0);
    
    for (j=0;j<m;j++) {
        for (i=0;i<n;i++) {
            assert(fabs(F[i+j*n]-F2[j+i*m])<1E-4);
        }
        assert(fabs(s[j]-s2[j])<1E-4);
    }
    printf("%s utest2 : OK\n",__FILE__);
}

/**
 * This tutorial demonstrates simple Usage of RTKLIB over the ROS system.
 */
int main(int argc, char **argv)
{
  ros::init(argc, argv, "testRTKLIB");

  ros::NodeHandle n;

  ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);

  ros::Rate loop_rate(10);

  int count = 0;
  Eigen::MatrixXd testMatrix;

  while (ros::ok())
  {
    /**
     * This is a message object. You stuff it with data, and then publish it.
     */
    std_msgs::String msg;

    std::stringstream ss;
    ss << "hello world " << count;
    msg.data = ss.str();

    ROS_INFO("%s", msg.data.c_str());

    utest1();
    utest2();

    chatter_pub.publish(msg);

    ros::spinOnce();

    loop_rate.sleep();
    ++count;
  }


  return 0;
}