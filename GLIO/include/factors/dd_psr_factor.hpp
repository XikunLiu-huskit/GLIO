// #define D2R 3.1415926/180.0
#include <nlosExclusion/GNSS_Raw_Array.h>
// google implements commandline flags processing.
#include <gflags/gflags.h>
// google loging tools
#include <glog/logging.h>
#include <gnss_comm/gnss_constant.hpp>
#include <gnss_comm/gnss_utility.hpp>

using namespace gnss_comm;

#define psr_size_20 20 // 20 psr measurements with 9 DD measurements

// DD pseudorange with 20 pseudorange measurements
class dd_psr_factor_20: public ceres::SizedCostFunction<psr_size_20-1, 3, 3, 1, 3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    dd_psr_factor_20() = delete;
    dd_psr_factor_20(const nlosExclusion::GNSS_Raw_Array &_gnss_data, const nlosExclusion::GNSS_Raw_Array &_ref_gnss_data, const Eigen::MatrixXd &_DD_W_matrix,const int &_mPrn, const double &_ratio, const Eigen::Vector3d &_Station_pos, const double &_DDpsrThreshold):
    gnss_data(_gnss_data),ref_gnss_data(_ref_gnss_data), DD_W_matrix(_DD_W_matrix),mPrn(_mPrn), ratio(_ratio), Station_pos(_Station_pos), DDpsrThreshold(_DDpsrThreshold)
    {
        relative_sqrt_info = 10.0;
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);

        double yaw_diff = parameters[2][0]; // yaw difference between enu and local
        Eigen::Vector3d ref_ecef(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Vector3d lever_arm_T = Eigen::Vector3d::Zero();
        double sin_yaw_diff = std::sin(yaw_diff);
        double cos_yaw_diff = std::cos(yaw_diff);
        Eigen::Matrix3d R_enu_local; // rotation of local to enu
        R_enu_local << cos_yaw_diff, -sin_yaw_diff, 0,
                sin_yaw_diff,  cos_yaw_diff, 0,
                0           ,  0           , 1;
        Eigen::Matrix3d R_ecef_enu = ecef2rotation(ref_ecef); // enu to ecef rotation
        Eigen::Matrix3d R_ecef_local = R_ecef_enu * R_enu_local; // local to ecef

        const Eigen::Vector3d local_pos_ = ratio*Pi + (1.0 - ratio)*Pj; // itepolation
        const Eigen::Vector3d local_pos = local_pos_ + lever_arm_T;    // lever arm correction

        Eigen::Vector3d P_ecef = R_ecef_local * local_pos + ref_ecef; // pose in ecef

        /* DD related */
        Eigen::Vector3d sv_pos;
        Eigen::Vector3d u_m_sv_pos; // user rcv to master satellite
        Eigen::Vector3d r_m_sv_pos; // reference to master satellite
        Eigen::Vector3d r_i_sv_pos; // reference to ith satellite
        Eigen::Vector3d station_pos; // master satellite
        /* DD related: base position */
        station_pos = Eigen::Vector3d(Station_pos[0], Station_pos[1], Station_pos[2]);

        int sat_num = gnss_data.GNSS_Raws.size();
        Eigen::Matrix<double, psr_size_20-1, psr_size_20-1> DD_W_matrix_ep;
        DD_W_matrix_ep.setZero();
        DD_W_matrix_ep.block(0, 0, sat_num-1, sat_num-1) = DD_W_matrix;

        int res_index = 0;
        for(int i = 0; i < sat_num; i++)
        {
            if(i!=mPrn)
            {
                sv_pos = Eigen::Vector3d(gnss_data.GNSS_Raws[i].sat_pos_x, gnss_data.GNSS_Raws[i].sat_pos_y, gnss_data.GNSS_Raws[i].sat_pos_z); // user to ith satellite sat pose

                u_m_sv_pos = Eigen::Vector3d(gnss_data.GNSS_Raws[mPrn].sat_pos_x, gnss_data.GNSS_Raws[mPrn].sat_pos_y, gnss_data.GNSS_Raws[mPrn].sat_pos_z);

                r_m_sv_pos = Eigen::Vector3d(ref_gnss_data.GNSS_Raws[mPrn].sat_pos_x, ref_gnss_data.GNSS_Raws[mPrn].sat_pos_y, ref_gnss_data.GNSS_Raws[mPrn].sat_pos_z);

                r_i_sv_pos = Eigen::Vector3d(ref_gnss_data.GNSS_Raws[i].sat_pos_x, ref_gnss_data.GNSS_Raws[i].sat_pos_y, ref_gnss_data.GNSS_Raws[i].sat_pos_z);

                /* line-of-sight vector user rcv-> ith satellite */
                Eigen::Vector3d rcv2sat_ecef = sv_pos - P_ecef;
                Eigen::Vector3d rcv2sat_unit = rcv2sat_ecef.normalized();

                /* line-of-sight vector user rcv-> master satellite */
                Eigen::Vector3d rcv2sat_ecef_u2m = u_m_sv_pos - P_ecef;
                Eigen::Vector3d rcv2sat_unit_u2m = rcv2sat_ecef_u2m.normalized();

                /* line-of-sight vector reference station-> ith satellite */
                Eigen::Vector3d rcv2sat_ecef_r2i = r_i_sv_pos - station_pos;
                Eigen::Vector3d rcv2sat_unit_r2i = rcv2sat_ecef_r2i.normalized();

                /* line-of-sight vector reference station-> master satellite */
                Eigen::Vector3d rcv2sat_ecef_r2m = r_m_sv_pos - station_pos;
                Eigen::Vector3d rcv2sat_unit_r2m = rcv2sat_ecef_r2m.normalized();

                double psr_estimated_u2i = rcv2sat_ecef.norm();
                double psr_estimated_u2m = rcv2sat_ecef_u2m.norm();
                double psr_estimated_r2i = rcv2sat_ecef_r2i.norm();
                double psr_estimated_r2m = rcv2sat_ecef_r2m.norm();

                double DD_psr_estimated = (psr_estimated_u2i - psr_estimated_r2i) - (psr_estimated_u2m - psr_estimated_r2m);

                double DD_psr = (gnss_data.GNSS_Raws[i].raw_pseudorange - ref_gnss_data.GNSS_Raws[i].raw_pseudorange) - (gnss_data.GNSS_Raws[mPrn].raw_pseudorange - ref_gnss_data.GNSS_Raws[mPrn].raw_pseudorange);

                double DD_psr_weight = 1.0;
                residuals[res_index] = DD_psr_weight* (DD_psr_estimated - DD_psr);
                if (fabs(residuals[res_index]) > DDpsrThreshold) DD_psr_weight = 0.05*DD_psr_weight;
                residuals[res_index] = DD_psr_weight* (DD_psr_estimated - DD_psr);

                if (jacobians)
                {
                    // J_Pi
                    if (jacobians[0])
                    {
                        Eigen::Map<Eigen::Matrix<double, psr_size_20-1, 3, Eigen::RowMajor>> J_Pi(jacobians[0]);

                        J_Pi.block<1,3>(res_index,0) = (-rcv2sat_unit.transpose() * R_ecef_local * DD_psr_weight * ratio)  - (-rcv2sat_unit_u2m.transpose() * R_ecef_local * DD_psr_weight * ratio);
                    }
                    // J_Pj
                    if (jacobians[1])
                    {
                        Eigen::Map<Eigen::Matrix<double, psr_size_20-1, 3, Eigen::RowMajor>> J_Pj(jacobians[1]);

                        J_Pj.block<1,3>(res_index,0) = (-rcv2sat_unit.transpose() * R_ecef_local * DD_psr_weight * (1.0 - ratio))  - (-rcv2sat_unit_u2m.transpose() * R_ecef_local * DD_psr_weight * (1.0 - ratio));
                    }
                }

                // increase the residual index
                res_index++;
            }
        }
        while (res_index < psr_size_20 - 1) {
            residuals[res_index] = 0;

            if (jacobians)
            {
                // J_Pi
                if (jacobians[0])
                {
                    Eigen::Map<Eigen::Matrix<double, psr_size_20-1, 3, Eigen::RowMajor>> J_Pi(jacobians[0]);

                    J_Pi.block<1,3>(res_index,0).setZero();
                }
                // J_Pj
                if (jacobians[1])
                {
                    Eigen::Map<Eigen::Matrix<double, psr_size_20-1, 3, Eigen::RowMajor>> J_Pj(jacobians[1]);

                    J_Pj.block<1,3>(res_index,0).setZero();
                }
            }

            // increase the residual index
            res_index++;
        }

        Eigen::Map<Eigen::Matrix<double, psr_size_20-1, 1>> residual(residuals);
        residual = DD_W_matrix_ep * residual;

        if (jacobians)
        {
            // J_Pi
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, psr_size_20-1, 3, Eigen::RowMajor>> J_PiTmp(jacobians[0]);
                J_PiTmp = DD_W_matrix_ep * J_PiTmp;
            }
            // J_Pj
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, psr_size_20-1, 3, Eigen::RowMajor>> J_PjTmp(jacobians[1]);
                J_PjTmp = DD_W_matrix_ep * J_PjTmp;
            }
        }

        return true;
    }
    bool check_gradients(const std::vector<const double*> &parameters) const;
private:
    double ratio;
    int freq_idx;
    double freq;

    Eigen::Vector3d sv_vel;
    double svdt, svddt, tgd;
    double pr_uura, dp_uura;
    double relative_sqrt_info, DDpsrThreshold;
    Eigen::Vector3d ecefRefP, Station_pos;
    int satSys;
    const nlosExclusion::GNSS_Raw_Array gnss_data;
    const nlosExclusion::GNSS_Raw_Array ref_gnss_data;

    Eigen::MatrixXd DD_W_matrix;
    int mPrn;
};
