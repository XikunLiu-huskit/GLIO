FROM osrf/ros:melodic-desktop-full

# ENV CATKIN_WS=/root/catkin_ws

RUN   apt-get update \ 
      && apt-get install -y \
      cmake \
      libatlas-base-dev \
      #libeigen3-dev \
      libgoogle-glog-dev \
      #libogre-1.9-dev\
      #libsuitesparse-dev \
      #python-catkin-tools \
      ros-melodic-novatel-msgs \
      software-properties-common \
      && add-apt-repository -y ppa:borglab/gtsam-release-4.0 \
      && apt-get update \
      && apt install -y libgtsam-dev libgtsam-unstable-dev \
      && rm -rf /var/lib/apt/lists/*
      
SHELL ["/bin/bash", "-c"]
      
RUN   mkdir -p ~/software_ws \
      && mkdir -p ~/catkin_ws/src \
      && cd ~/catkin_ws/src \
      && git clone https://github.com/XikunLiu-huskit/GLIO.git \
      && mv ./GLIO/support_files/ceres-solver.tar.gz ~/software_ws/ \
      && cd ~/software_ws \
      && tar -zxvf ceres-solver.tar.gz \
      && mkdir -p ceres-bin \
      && cd ceres-bin \
      && cmake ../ceres-solver \
      && make install \ 
      && cd ~/catkin_ws \
      && source /opt/ros/melodic/setup.bash \
      && catkin_make
      
RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc \
    && echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

WORKDIR /root/catkin_ws
