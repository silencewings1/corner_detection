#include <opencv2/opencv.hpp>
#include "alg/math_helper.h"
#include "alg/timer.h"
#include "detector/detector.h"
#include "detector/rectifier.h"
#include <iostream>
using namespace cv;
using namespace std;


void test_refy()
{
	String image_name_left = "../../imgs_1g3p_4_line_60mm/left/left_1.png";
	Mat image_left = imread(image_name_left);
	String image_name_right = "../../imgs_1g3p_4_line_60mm/right/right_1.png";
	Mat image_right = imread(image_name_right);

	Rectifier rectifier(cv::Size(1920, 1080));
	auto img_after_left = rectifier.rectify(image_left, Rectifier::LEFT);
	auto img_after_right = rectifier.rectify(image_right, Rectifier::RIGHT);

	imshow("left_ori", image_left);
	imshow("left_after", img_after_left);

	imshow("right_ori", image_right);
	imshow("right_after", img_after_right);

	imwrite("../../img_res/left_after.png", img_after_left);
	imwrite("../../img_res/right_after.png", img_after_right);

	waitKey(0);
}

void test_whole()
{
	cv::String image_name = "../../imgs_1g3p_4_line_60mm/left/left_1.png";
	cv::Mat image = imread(image_name);
	if (!image.data)
	{
		printf(" No image data \n ");
		return;
	}

	Rectifier rectifier(cv::Size(1920, 1080));
	auto t0 = tic();
	auto img_rectified = rectifier.rectify(image, Rectifier::LEFT);
	toc(t0, "t0");

	cv::imshow("img_rectified", img_rectified);

	Detector detector(img_rectified.size());
	detector.findCorners(img_rectified);
}

int main()
{
	////////////////
	//cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
	////////////////

	String image_name = "../../imgs_1g3p_4_line_60mm/left/left_1_rectified.png";
	Mat image = imread(image_name);
	if (!image.data)
	{
		printf(" No image data \n ");
		return -1;
	}

	Detector detector(image.size());
	auto total = tic();
	auto corners = detector.findCorners(image);
	toc(total, "total");
	detector.showResult(corners, image);

	waitKey(0);
	return 0;
}