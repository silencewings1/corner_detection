#include "config.h"

Eigen::Matrix3d A_cam;
double b_dis;
double OPTI_COST_THRESHOLD;

bool loadConfig()
{
    A_cam << 1156.3, 0, 1000.5,
        0, 1156.3, 560.3,
        0, 0, 1;
    b_dis = 4.0496;

    OPTI_COST_THRESHOLD = 10.0;

    return true;
}
