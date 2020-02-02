//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//int main()
//{
//	Mat image = Mat::zeros(300, 600, CV_8UC3);
//	circle(image, Point(250, 150), 100, Scalar(0, 255, 128), -100);
//	circle(image, Point(350, 150), 100, Scalar(255, 255, 255), -100);
//	imshow("Display Window", image);
//	waitKey(0);
//	return 0;
//}

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
	Mat B = (Mat_<float>(2, 2) << 0.8000, 0.1000, 0.3000, 0.5000);
	Mat C = conv2(A, B, "valid");
	Mat D = conv2(A, B, "same");
	Mat E = conv2(A, B, "full");

	cout.setf(ios::fixed);
	cout << "valid:" << endl
		 << C << endl
		 << endl
		 << "full:" << endl
		 << E << endl
		 << endl
		 << "same:" << endl
		 << D << endl;

	Mat ker = (Mat_<float>(1, 3) << -1, 0, 1);
	Mat kk = conv2(A, ker, "same");
	cout << "kernel:" << endl
		 << kk << endl
		 << endl;
}

void test_refy()
{
	String image_name = "../../imgs_1g3p_4_line_60mm/left/left_1.png";
	String image_name_2 = "../../imgs_1g3p_4_line_60mm/left/left_1_rectified.png";
	Mat image = imread(image_name);
	Mat image_2 = imread(image_name_2);

	Rectifier rectifier(cv::Size(1920, 1080));
	auto img_after = rectifier.rectify(image, Rectifier::LEFT);
	auto sub = image - img_after;
	auto sub2 = image_2 - img_after;

	imshow("ori", image);
	imshow("after", img_after);
	imshow("sub", sub);
	imshow("sub2", sub2);
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
	//test_conv();

	// test_whole();
	// waitKey(0);
	// return 0;

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