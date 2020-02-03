#include <opencv2/opencv.hpp>
#include "alg/math_helper.h"
#include "detector/detector.h"
#include "detector/rectifier.h"
#include <iostream>
using namespace cv;
using namespace std;

void test_conv()
{
	Mat A = (Mat_<float>(6, 6) << 12.8000, 3.2000, -7.0000, 9.5000, -0.5000, -7.8000,
			 8.8000, 17.9000, 19.2000, 7.6000, 3.4000, 4.3000,
			 2.3000, 11.5000, 7.7000, 15.5000, 8.0000, -4.2000,
			 3.9000, 13.8000, 17.6000, -0.0000, -0.7000, 6.2000,
			 -2.4000, 3.4000, 10.1000, 11.4000, 9.8000, -1.7000,
			 -1.6000, -5.6000, -6.8000, -3.2000, -3.0000, -0.2000);
	//Mat B = (Mat_<float>(3, 3) << 0.8000, 0.1000, -0.6000, 0.3000, 0.5000, 0.7000, -0.4000, 0, -0.2000);

	String image_name_left = "../../imgs_1g3p_4_line_60mm/left/left_1.png";
	Mat AA = imread(image_name_left);
	cv::cvtColor(AA, A, cv::COLOR_BGR2GRAY);
	[&](cv::Mat &img) {
		img.convertTo(img, MatType);
		img = img / 255;
	}(A);

	Mat B = (Mat_<float>(2, 2) << 0.8000, 0.1000, 0.3000, 0.5000);
	Mat C = conv2(A, B, "valid");
	Mat D = conv2(A, B, "same");

	Mat E; // = conv2(A, B, "full");
	auto te = tic();
	for (int i = 0; i < 10; ++i)
		E = conv2(A, B, "full");
	toc(te, "te");

	Mat F; // = cudaFilter(A, B);
	auto tf = tic();
	for (int i = 0; i < 10; ++i)
		F = cudaFilter(A, B);
	toc(tf, "tf");

	//cout.setf(ios::fixed);
	// cout//  << "valid:" << endl
	// 	//  << C << endl
	// 	//  << endl
	// 	//  << "full:" << endl
	// 	//  << E << endl
	// 	//  << endl
	// 	 << "same:" << endl
	// 	 << D << endl
	// 	 << endl
	// 	 << "cuda:" << endl
	// 	 << F << endl;

	// Mat ker = (Mat_<float>(1, 3) << -1, 0, 1);
	// Mat kk = conv2(A, ker, "same");
	// cout << "kernel:" << endl
	// 	 << kk << endl
	// 	 << endl;
}

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

	Detector detector(img_rectified);
}

int main()
{
	////////////////
	cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
	////////////////

	String image_name = "../../imgs_1g3p_4_line_60mm/left/left_1_rectified.png";
	//image_name = "D:/Research/Vision/liqi/imgs_1g3p_4_line_60mm/left/sss.png";
	//image_name = "D:/Research/github_corner_detect/cornerDetection/02.png";
	Mat image = imread(image_name);
	if (!image.data)
	{
		printf(" No image data \n ");
		return -1;
	}

	Detector detector(image);

	waitKey(0);
	return 0;
}