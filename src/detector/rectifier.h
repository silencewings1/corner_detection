#pragma once
#include "opencv2/opencv.hpp"

class Rectifier
{
public:
	Rectifier(const cv::Size& img_size)
	{
		init_rectify_para(img_size);
	}

	enum ImgIdx { LEFT, RIGHT };
	cv::Mat rectify(const cv::Mat& img, const ImgIdx& id) const
	{
		cv::Mat res;
		switch (id)
		{
		case ImgIdx::LEFT: cv::remap(img, res, left_map1, left_map2, cv::INTER_LINEAR);
		case ImgIdx::RIGHT: cv::remap(img, res, right_map1, right_map2, cv::INTER_LINEAR);
		}

		return res;
	}

private:
	void init_rectify_para(const cv::Size& img_size)
	{
		cv::Mat left_camera_matrix = (cv::Mat_<double>(3, 3)
			<< 1096.6, 0, 1004.6,
			0, 1101.2, 547.8316,
			0, 0, 1);
		cv::Mat left_distortion = (cv::Mat_<double>(1, 5)
			<< 0.0757, -0.0860, -2.2134e-4, 2.4925e-4, 0.00000);

		cv::Mat right_camera_matrix = (cv::Mat_<double>(3, 3)
			<< 1095.2, 0, 996.3653,
			0, 1100.7, 572.7941,
			0, 0, 1);
		cv::Mat right_distortion = (cv::Mat_<double>(1, 5)
			<< 0.0775, -0.0932, -9.2589e-4, 1.5443e-4, 0.00000);

		cv::Mat R = (cv::Mat_<double>(3, 3)
			<< 0.999991605633247, -0.000345422533180, 0.004082811079910,
			0.000333318919503, 0.999995549306528, 0.002964838213577,
			-0.004083817030495, -0.002963452447460, 0.999987270113001);
		cv::Mat t = (cv::Mat_<double>(3, 1)
			<< -4.049079591541234, -0.004234103893078, -0.065436975906188);


		cv::Mat R1, R2, P1, P2, Q;
		cv::stereoRectify(
			left_camera_matrix, left_distortion, right_camera_matrix, right_distortion, img_size, R, t,
			R1, R2, P1, P2, Q);

		cv::initUndistortRectifyMap(
			left_camera_matrix, left_distortion, R1, P1, img_size, CV_16SC2,
			left_map1, left_map2);

		cv::initUndistortRectifyMap(
			right_camera_matrix, right_distortion, R2, P2, img_size, CV_16SC2,
			right_map1, right_map2);
	}

private:
	cv::Mat left_map1;
	cv::Mat left_map2;
	cv::Mat right_map1;
	cv::Mat right_map2;
};