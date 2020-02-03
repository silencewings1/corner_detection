#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
#include <Eigen/Core>


using PixelType = float;
const PixelType PI = CV_PI;
const int MatType = CV_32FC1;


struct Maxima
{
	Maxima(int i, int j, PixelType val)
		: corner(i, j), val(val)
	{}

	cv::Point corner;
	PixelType val;
};
using Maximas = std::vector<Maxima>;

using Orientation = Eigen::Matrix<PixelType, 2, 1>;

struct Angle
{
	Orientation v1;
	Orientation v2;
	PixelType avg_angle;
};
using Angles = std::vector<Angle>;

using Point2p = cv::Point_<PixelType>;
struct Corner
{
	Corner(const Point2p& point, const Angle& angle)
		: point(point), angle(angle)
	{}
	Point2p point;
	Angle angle;
};
using Corners = std::vector<Corner>;

struct ScoreCorner
{
	Corner corner; 
	PixelType score;
};
using ScoreCorners = std::vector<ScoreCorner>;