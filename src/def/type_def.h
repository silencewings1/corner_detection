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

using Corner = cv::Point_<PixelType>;
using Corners = std::vector<Corner>;

struct CornerTemplate
{
	CornerTemplate(const Corner& point, int width)
		: point(point), width(width), angle(0), corr(0)
	{}

	Corner point;
	PixelType width;
	PixelType angle;
	PixelType corr;
};
using CornersTemplate = std::vector<CornerTemplate>;

struct DetectRectangle
{
	bool is_full(const cv::Size& size) const
	{
		if (range_x.start == 0 && range_x.end == size.width &&
			range_y.start == 0 && range_y.end == size.height)
		{
			return true;
		}

		return false;
	}

	cv::Range range_x;
	cv::Range range_y;
};