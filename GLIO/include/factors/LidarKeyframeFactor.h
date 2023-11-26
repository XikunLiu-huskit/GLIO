#ifndef LIDARFACTOR_H
#define LIDARFACTOR_H

#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <assert.h>
#include <cmath>
#include "utils/math_tools.h"

struct LidarEdgeFactor
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LidarEdgeFactor(Eigen::Vector3d curr_point_,
                    Eigen::Vector3d last_point_a_,
                    Eigen::Vector3d last_point_b_,
                    Eigen::Quaterniond qlb_,
                    Eigen::Vector3d tlb_,
                    double s_)
        : curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_) {
        qlb = qlb_;
        tlb = tlb_;
        s = s_;
    }

    template <typename T>
    bool operator()(const T *t, const T *q, T *residual) const
    {

        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
        Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

        Eigen::Quaternion<T> q_last_curr{q[0], q[1], q[2], q[3]};

        Eigen::Matrix<T, 3, 1> t_last_curr{t[0], t[1], t[2]};

        Eigen::Quaternion<T> q_l_b{T(qlb.w()), T(qlb.x()), T(qlb.y()), T(qlb.z())};
        Eigen::Matrix<T, 3, 1> t_l_b{T(tlb.x()), T(tlb.y()), T(tlb.z())};

        Eigen::Matrix<T, 3, 1> lp;
        lp = q_l_b.inverse() * (cp - t_l_b);
        lp = q_last_curr * lp + t_last_curr;

        Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
        Eigen::Matrix<T, 3, 1> de = lpa - lpb;

        residual[0] = nu.norm() / de.norm();
        residual[0] *= T(s);

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_,
                                       const Eigen::Vector3d last_point_a_,
                                       const Eigen::Vector3d last_point_b_,
                                       Eigen::Quaterniond qlb_,
                                       Eigen::Vector3d tlb_,
                                       double s_)
    {
        return (new ceres::AutoDiffCostFunction<LidarEdgeFactor, 1, 3, 4>(
                    new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, qlb_, tlb_, s_)));
    }

    Eigen::Vector3d curr_point, last_point_a, last_point_b;
    Eigen::Quaterniond qlb;
    Eigen::Vector3d tlb;
    double s;
};


struct LidarPlaneNormFactor
{
    LidarPlaneNormFactor(Eigen::Vector3d curr_point_,
                         Eigen::Vector3d plane_unit_norm_,
                         Eigen::Quaterniond qlb_,
                         Eigen::Vector3d tlb_,
                         double negative_OA_dot_norm_,
                         double score_) : curr_point(curr_point_),
        plane_unit_norm(plane_unit_norm_),
        qlb(qlb_),
        tlb(tlb_),
        negative_OA_dot_norm(negative_OA_dot_norm_),
        score(score_) {}

    template <typename T>
    bool operator()(const T *t, const T *q, T *residual) const
    {
        Eigen::Quaternion<T> q_w_curr{q[0], q[1], q[2], q[3]};
        Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> point_w;
        Eigen::Quaternion<T> q_l_b{T(qlb.w()), T(qlb.x()), T(qlb.y()), T(qlb.z())};
        Eigen::Matrix<T, 3, 1> t_l_b{T(tlb.x()), T(tlb.y()), T(tlb.z())};

        point_w = q_l_b.inverse() * (cp - t_l_b);
        point_w = q_w_curr * point_w + t_w_curr;

        Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
        residual[0] = T(score) * (norm.dot(point_w) + T(negative_OA_dot_norm));
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_,
                                       const Eigen::Vector3d plane_unit_norm_,
                                       const Eigen::Quaterniond qlb_,
                                       const Eigen::Vector3d tlb_,
                                       const double negative_OA_dot_norm_,
                                       const double score_)
    {
        return (new ceres::AutoDiffCostFunction<
                LidarPlaneNormFactor, 1, 3, 4>(
                    new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, qlb_, tlb_, negative_OA_dot_norm_, score_)));
    }

    Eigen::Vector3d curr_point;
    Eigen::Vector3d plane_unit_norm;
    Eigen::Quaterniond qlb;
    Eigen::Vector3d tlb;
    double negative_OA_dot_norm, score;
};

struct BinaryLidarPlaneNormFactor
{
    BinaryLidarPlaneNormFactor(Eigen::Vector3d curr_point_,
                         Eigen::Matrix<double, 6, 1> planet_norm_cent_,
                         double score_) : curr_point(curr_point_),
                                          planet_norm_cent(planet_norm_cent_),
                                          score(score_) {}

    template <typename T>
    bool operator()(const T *t1, const T *q1, const T *t2, const T *q2, T *residual) const
    {
        Eigen::Quaternion<T> q_w_curr{q1[0], q1[1], q1[2], q1[3]};
        Eigen::Matrix<T, 3, 1> t_w_curr{t1[0], t1[1], t1[2]};
        Eigen::Quaternion<T> q_w_oth{q2[0], q2[1], q2[2], q2[3]};
        Eigen::Matrix<T, 3, 1> t_w_oth{t2[0], t2[1], t2[2]};
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 6, 1> pnc = planet_norm_cent.cast<T>();

        Eigen::Matrix<T, 3, 1> point_w, normal_oth, cent_oth;

        point_w = q_w_curr * cp + t_w_curr;
        normal_oth = q_w_oth * pnc.head(3);
        cent_oth = q_w_oth * pnc.tail(3) + t_w_oth;

        residual[0] = T(score) * (normal_oth.dot(point_w - cent_oth));
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_,
                                       const Eigen::Matrix<double, 6, 1> planet_norm_cent_,
                                       const double score_)
    {
        return (new ceres::AutoDiffCostFunction<
                BinaryLidarPlaneNormFactor, 1, 3, 4, 3, 4>(
                new BinaryLidarPlaneNormFactor(curr_point_, planet_norm_cent_, score_)));
    }

    Eigen::Vector3d curr_point;
    Eigen::Matrix<double, 6, 1> planet_norm_cent;
    double score;
};

struct LidarPlaneNormEulerFactor
{
    LidarPlaneNormEulerFactor(Eigen::Vector3d curr_point_,
                             Eigen::Vector3d plane_unit_norm_,
                             Eigen::Quaterniond qlb_,
                             Eigen::Vector3d tlb_,
                             double negative_OA_dot_norm_,
                             double score_) : curr_point(curr_point_),
        plane_unit_norm(plane_unit_norm_),
        qlb(qlb_),
        tlb(tlb_),
        negative_OA_dot_norm(negative_OA_dot_norm_),
        score(score_) {}

    template <typename T>
    bool operator()(const T *t, const T *e, T *residual) const
    {
        Eigen::Quaternion<T> q_w_curr;
        Eigen::Matrix<T, 3, 1> e_{e[0], e[1], e[2]};
        q_w_curr = Eigen::AngleAxis<T>(e_[0], Eigen::Matrix<T, 3, 1>::UnitX()) * 
                    Eigen::AngleAxis<T>(e_[1], Eigen::Matrix<T, 3, 1>::UnitY()) * 
                    Eigen::AngleAxis<T>(e_[2], Eigen::Matrix<T, 3, 1>::UnitZ());
        Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> point_w;
        Eigen::Quaternion<T> q_l_b{T(qlb.w()), T(qlb.x()), T(qlb.y()), T(qlb.z())};
        Eigen::Matrix<T, 3, 1> t_l_b{T(tlb.x()), T(tlb.y()), T(tlb.z())};

        point_w = q_l_b.inverse() * (cp - t_l_b);
        point_w = q_w_curr * point_w + t_w_curr;

        Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
        residual[0] = T(score) * (norm.dot(point_w) + T(negative_OA_dot_norm));
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_,
                                       const Eigen::Vector3d plane_unit_norm_,
                                       const Eigen::Quaterniond qlb_,
                                       const Eigen::Vector3d tlb_,
                                       const double negative_OA_dot_norm_,
                                       const double score_)
    {
        return (new ceres::AutoDiffCostFunction<
                LidarPlaneNormEulerFactor, 1, 3, 3>(
                    new LidarPlaneNormEulerFactor(curr_point_, plane_unit_norm_, qlb_, tlb_, negative_OA_dot_norm_, score_)));
    }

    Eigen::Vector3d curr_point;
    Eigen::Vector3d plane_unit_norm;
    Eigen::Quaterniond qlb;
    Eigen::Vector3d tlb;
    double negative_OA_dot_norm, score;
};


struct LidarPlaneNormIncreFactor
{

    LidarPlaneNormIncreFactor(Eigen::Vector3d curr_point_,
                              Eigen::Vector3d plane_unit_norm_,
                              double negative_OA_dot_norm_)
        : curr_point(curr_point_),
          plane_unit_norm(plane_unit_norm_),
          negative_OA_dot_norm(negative_OA_dot_norm_) {}

    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {
        Eigen::Quaternion<T> q_inc{q[0], q[1], q[2], q[3]};
        Eigen::Matrix<T, 3, 1> t_inc{t[0], t[1], t[2]};
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> point_w;
        point_w = q_inc * cp + t_inc;

        Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
        residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_,
                                       const Eigen::Vector3d plane_unit_norm_,
                                       const double negative_OA_dot_norm_)
    {
        return (new ceres::AutoDiffCostFunction<LidarPlaneNormIncreFactor, 1, 4, 3>(
                    new LidarPlaneNormIncreFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
    }

    Eigen::Vector3d curr_point;
    Eigen::Vector3d plane_unit_norm;
    double negative_OA_dot_norm;
};

#if 1
/* roll pitch factor*/
struct roll_pitch_factor_auto
{
    roll_pitch_factor_auto(const Eigen::Vector3d &up_point_vect)
    :up_point_vect(up_point_vect){}

    template <typename T>
    bool operator()(const T* quat_, T* residuals) const
    {
        Eigen::Quaternion<T> quat{T(quat_[0]), T(quat_[1]), T(quat_[2]), T(quat_[3])};
        Eigen::Matrix<T, 3, 1> up_point_vect_tmp = quat * Eigen::Vector3d(0, 0, 1).cast<T>();

        residuals[0] = T(20) * (T(1) - up_point_vect.dot(up_point_vect_tmp));

        return true;
    }

    Eigen::Vector3d up_point_vect;

};

struct delta_q_factor_auto
{
    delta_q_factor_auto(const Eigen::Quaterniond &delta_q)
            :delta_q(delta_q){}

    template <typename T>
    bool operator()(const T* quat_i, const T* quat_j, T* residuals) const
    {
        Eigen::Quaternion<T> quati{T(quat_i[0]), T(quat_i[1]), T(quat_i[2]), T(quat_i[3])};
        Eigen::Quaternion<T> quatj{T(quat_j[0]), T(quat_j[1]), T(quat_j[2]), T(quat_j[3])};
        Eigen::Quaternion<T> tmp_delta_q(T(delta_q.w()), T(delta_q.x()), T(delta_q.y()), T(delta_q.z()));

        Eigen::Matrix<T, 3, 1> residual = T(10000) * (tmp_delta_q.inverse() * quati.inverse() * quatj).vec();

        residuals[0] = residual[0];
        residuals[1] = residual[1];
        residuals[2] = residual[2];

        return true;
    }

    Eigen::Quaterniond delta_q;

};
#endif

#endif // LIDARFACTOR_H
