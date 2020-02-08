#pragma once
#include <Eigen/Core>

extern Eigen::Matrix3d A_cam;
extern double b_dis;
extern double OPTI_COST_THRESHOLD;

bool loadConfig();