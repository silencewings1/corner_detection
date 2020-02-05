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

void test_image()
{
	String image_name = "../../imgs_1g3p_4_line_60mm/left/left_1_rectified.png";
	Mat image = imread(image_name);
	if (!image.data)
	{
		printf(" No image data \n ");
		return;
	}

	Detector detector(image.size());
	auto total = tic();
	auto corners = detector.process(image);
	toc(total, "total");
	detector.showResult(corners, image);

	waitKey(0);
}

void test_video()
{
	VideoCapture capture;
	capture.open("../../video20200120/WIN_20200120_11_39_46_Pro.mp4");
	if (!capture.isOpened())
	{
		printf("can not open ...\n");
		return ;
	}

	Mat frame;
	while (capture.read(frame))
	{
		Detector detector(frame.size());
		auto total = tic();
		auto corners = detector.process(frame);
		toc(total, "total");
		detector.showResult(corners, frame);

		auto key = waitKey(1);
		if (key == 'q' || key == 'Q')
			break;
	}

	capture.release();
}

int main()
{
	////////////////
	//cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
	////////////////

	test_video();
	return 0;
}