
#ifndef INC_3DLA_GLIO_DOPP_FACTOR_H
#define INC_3DLA_GLIO_DOPP_FACTOR_H
// #define D2R 3.1415926/180.0
#include <nlosExclusion/GNSS_Raw_Array.h>
// google implements commandline flags processing.
#include <gflags/gflags.h>
// google loging tools
#include <glog/logging.h>

#include <vector>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include <gnss_comm/gnss_constant.hpp>
#include <gnss_comm/gnss_utility.hpp>

/* tightly coupled doppler factor*/
struct tcdopplerFactor
{
    tcdopplerFactor(int epoch, std::string sat_sys, nlosExclusion::GNSS_Raw gnss_data, double ts_ratio, Eigen::Matrix<double, 3, 1> lever_arm_T, Eigen::Matrix3d R_ecef_local, double var)
            :epoch(epoch), sat_sys(sat_sys), gnss_data(gnss_data), ts_ratio(ts_ratio), lever_arm_T(lever_arm_T), R_ecef_local(R_ecef_local), var(var){}

    template <typename T>
    bool operator()(const T* statePi,const T* stateVi, const T* statePj,const T* stateVj,
            const T* para_rcv_ddt, const T* para_yaw_enu_local, const T* para_anc_ecef, T* residuals) const
    {
        Eigen::Vector3d sv_pos_ = Eigen::Vector3d(gnss_data.sat_pos_x,gnss_data.sat_pos_y,gnss_data.sat_pos_z);
        Eigen::Vector3d sv_vel_ = Eigen::Vector3d(gnss_data.vel_x, gnss_data.vel_y, gnss_data.vel_z);
        Eigen::Matrix<T, 3, 1> sv_pos = sv_pos_.cast<T>();
        Eigen::Matrix<T, 3, 1> sv_vel = sv_vel_.cast<T>();
        Eigen::Matrix<T, 3, 1> Pi(statePi[0], statePi[1], statePi[2]);
        Eigen::Matrix<T, 3, 1> Vi(stateVi[0], stateVi[1], stateVi[2]);
        Eigen::Matrix<T, 3, 1> Pj(statePj[0], statePj[1], statePj[2]);
        Eigen::Matrix<T, 3, 1> Vj(stateVj[0], stateVj[1], stateVj[2]);
        T svddt = T (gnss_data.ddt);

        T rcv_ddt = para_rcv_ddt[epoch];
//        if (sat_sys == "GPS") {
//            rcv_ddt = para_rcv_ddt[0]; // receiver clock bias drift rate
//        }
//        else if (sat_sys == "BDS") {
//            rcv_ddt = para_rcv_ddt[1]; // receiver clock bias drift rate
//        }
//        else if (sat_sys == "GLO") {
//            rcv_ddt = para_rcv_ddt[2]; // receiver clock bias drift rate
//        }
//        else if (sat_sys == "GAL") {
//            rcv_ddt = para_rcv_ddt[3]; // receiver clock bias drift rate
//        }

        const Eigen::Matrix<T, 3, 1> local_pos_ = T(ts_ratio)*Pi + (T(1.0) - T(ts_ratio))*Pj; // itepolation
        const Eigen::Matrix<T, 3, 1> local_pos = local_pos_ + lever_arm_T.cast<T>();    // lever arm correction
        const Eigen::Matrix<T, 3, 1> local_vel = T(ts_ratio)*Vi + (T(1.0) - T(ts_ratio))*Vj; // itepolation

        Eigen::Matrix<T, 3, 1> ref_ecef(para_anc_ecef[0], para_anc_ecef[1], para_anc_ecef[2]);
        Eigen::Matrix<T, 3, 1> P_ecef = R_ecef_local.cast<T>() * local_pos + ref_ecef; // pose in ecef
        Eigen::Matrix<T, 3, 1> V_ecef = R_ecef_local.cast<T>() * local_vel; // velocity in ecef

        /* line-of-sight vector user rcv-> ith satellite */
        Eigen::Matrix<T, 3, 1> rcv2sat_ecef = sv_pos - P_ecef;
        Eigen::Matrix<T, 3, 1> rcv2sat_unit = rcv2sat_ecef.normalized();

        /* realistic effects */
        const T dopp_sagnac = EARTH_OMG_GPS/LIGHT_SPEED*(sv_vel(0)*P_ecef(1)+
                                                              sv_pos(0)*V_ecef(1) - sv_vel(1)*P_ecef(0) - sv_pos(1)*V_ecef(0));

        /* estimated doppler: h(x) */
        T dopp_estimated = (sv_vel - V_ecef).dot(rcv2sat_unit) + dopp_sagnac + rcv_ddt - svddt;

        // dp_weight = dp_weight * double(LiDARFeatureNum/GNSSNum);
        residuals[0] = (dopp_estimated + T (gnss_data.doppler*gnss_data.lamda)) / T(var);

        return true;
    }

    double var, ts_ratio;
    std::string sat_sys; // satellite system
    nlosExclusion::GNSS_Raw gnss_data;
    Eigen::Matrix<double, 3, 1> lever_arm_T;
    Eigen::Matrix3d R_ecef_local;
    double doppler;
    int epoch;

};

/* constant clock drift factor*/
struct constantClockDriftFactor
{
    constantClockDriftFactor(int i, int j)
            :i(i), j(j){}

    template <typename T>
    bool operator()(const T* rcvDDt, T* residuals) const
    {

        residuals[0] = (rcvDDt[i] - rcvDDt[j]);

        return true;
    }

    int i, j;
};

/*
**  Function: motion constraint with different uncertainty on xyz.
**  parameters[0]: position at time k
**  parameters[1]: position at time k+1
**  parameters[1]: velocity at time k
**  parameters[1]: velocity at time k+1
**  residual[0]: the integer ambiguity value VS P_k
*/
class AnalyticalMotionModelFactor : public ceres::SizedCostFunction<3, 3, 3, 9, 9>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    AnalyticalMotionModelFactor() = delete;
    AnalyticalMotionModelFactor( const double _dt, const Eigen::MatrixXd &_W_matrix):  dt(_dt), W_matrix(_W_matrix)
    {

    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);

        Eigen::Vector3d Vi(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);

        // est_pseudorange = sqrt(delta_x+ delta_y + delta_z);

        residuals[0] = (Pj[0] - Pi[0])/ dt - ((Vi[0]+Vj[0])/(2));
        residuals[1] = (Pj[1] - Pi[1])/ dt - ((Vi[1]+Vj[1])/(2));
        residuals[2] = (Pj[2] - Pi[2])/ dt - ((Vi[2]+Vj[2])/(2));

        if (jacobians)
        {
            // J_Pi
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_Pi(jacobians[0]);
                J_Pi.setZero();

                J_Pi(0,0) = -1.0 / dt;
                J_Pi(1,1) = -1.0 / dt;
                J_Pi(2,2) = -1.0 / dt;

                J_Pi = W_matrix * J_Pi;
            }

            // J_Pj
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_Pj(jacobians[1]);
                J_Pj.setZero();

                J_Pj(0,0) = 1.0 / dt;
                J_Pj(1,1) = 1.0 / dt;
                J_Pj(2,2) = 1.0 / dt;

                J_Pj = W_matrix * J_Pj;
            }

            // J_Vi
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>> J_Vi(jacobians[2]);
                J_Vi.setZero();

                J_Vi(0,0) = -1.0/2.0;
                J_Vi(1,1) = -1.0/2.0;
                J_Vi(2,2) = -1.0/2.0;

                J_Vi = W_matrix * J_Vi;
            }

            // J_Vj
            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>> J_Vj(jacobians[3]);
                J_Vj.setZero();

                J_Vj(0,0) = -1.0/2.0;
                J_Vj(1,1) = -1.0/2.0;
                J_Vj(2,2) = -1.0/2.0;

                J_Vj = W_matrix * J_Vj;
            }

            Eigen::Map<Eigen::Matrix<double, 3, 1>> residual(residuals);
            residual = W_matrix * residual;


        }
        return true;
    }

    bool check_gradients(const std::vector<const double*> &parameters) const;
private:

    double dt;
    Eigen::MatrixXd W_matrix;
};

/* oriented doppler factor*/
struct orientedDopplerFactor
{
    orientedDopplerFactor(int epoch, std::string sat_sys, nlosExclusion::GNSS_Raw gnss_data, Eigen::Matrix3d R_ecef_local_origin2cur, Eigen::Vector3d p_ecef, Eigen::Vector3d v_local_trans2orgin, double var)
            :epoch(epoch), sat_sys(sat_sys), gnss_data(gnss_data), R_ecef_local_origin2cur(R_ecef_local_origin2cur), p_ecef(p_ecef), v_local_trans2orgin(v_local_trans2orgin), var(var){}

    template <typename T>
    bool operator()(const T* origin_quat, const T* para_rcv_ddt, T* residuals) const
    {
        Eigen::Quaternion<T> Q0(origin_quat[0], origin_quat[1], origin_quat[2], origin_quat[3]);
        T svddt = T (gnss_data.ddt);
        T rcv_ddt = para_rcv_ddt[epoch];
//        if (sat_sys == "GPS") {
//            rcv_ddt = para_rcv_ddt[0]; // receiver clock bias drift rate
//        }
//        else if (sat_sys == "BDS") {
//            rcv_ddt = para_rcv_ddt[1]; // receiver clock bias drift rate
//        }
//        else if (sat_sys == "GLO") {
//            rcv_ddt = para_rcv_ddt[2]; // receiver clock bias drift rate
//        }
//        else if (sat_sys == "GAL") {
//            rcv_ddt = para_rcv_ddt[3]; // receiver clock bias drift rate
//        }

        Eigen::Vector3d sv_pos_ = Eigen::Vector3d(gnss_data.sat_pos_x,gnss_data.sat_pos_y,gnss_data.sat_pos_z);
        Eigen::Vector3d sv_vel_ = Eigen::Vector3d(gnss_data.vel_x, gnss_data.vel_y, gnss_data.vel_z);
        Eigen::Matrix<T, 3, 1> sv_pos = sv_pos_.cast<T>();
        Eigen::Matrix<T, 3, 1> sv_vel = sv_vel_.cast<T>();

        Eigen::Matrix<T, 3, 1> P_ecef = p_ecef.cast<T>(); // pose in ecef
        Eigen::Matrix<T, 3, 1> V_ecef = R_ecef_local_origin2cur.cast<T>() * Q0 * v_local_trans2orgin; // velocity in ecef

        /* line-of-sight vector user rcv-> ith satellite */
        Eigen::Matrix<T, 3, 1> rcv2sat_ecef = sv_pos - P_ecef;
        Eigen::Matrix<T, 3, 1> rcv2sat_unit = rcv2sat_ecef.normalized();

        /* realistic effects */
        const T dopp_sagnac = EARTH_OMG_GPS/LIGHT_SPEED*(sv_vel(0)*P_ecef(1)+
                                                         sv_pos(0)*V_ecef(1) - sv_vel(1)*P_ecef(0) - sv_pos(1)*V_ecef(0));

        /* estimated doppler: h(x) */
        T dopp_estimated = (sv_vel - V_ecef).dot(rcv2sat_unit) + dopp_sagnac + rcv_ddt - svddt;

        // dp_weight = dp_weight * double(LiDARFeatureNum/GNSSNum);
        residuals[0] = (dopp_estimated + T (gnss_data.doppler*gnss_data.lamda)); // / T(var)

        return true;
    }

    double var, ts_ratio;
    std::string sat_sys; // satellite system
    nlosExclusion::GNSS_Raw gnss_data;
    Eigen::Matrix<double, 3, 1> lever_arm_T;
    Eigen::Matrix3d R_ecef_local_origin2cur;
    double doppler, vel_scale;
    int epoch;
    Eigen::Vector3d p_ecef;
    Eigen::Vector3d v_local_trans2orgin;

};

#endif //INC_3DLA_GLIO_DOPP_FACTOR_H
