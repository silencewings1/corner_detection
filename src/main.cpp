#include "alg/math_helper.h"
#include "alg/timer.h"
#include "detector/detector.h"
#include "detector/rectifier.h"
#include <thread>
#include <future>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
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
    detector.showResult("corners", corners, image);

    waitKey(0);
}

void test_video()
{
    VideoCapture capture;
    capture.open("../../video20200120/WIN_20200120_11_37_50_Pro.mp4");
    if (!capture.isOpened())
    {
        printf("can not open ...\n");
        return;
    }

    auto avg_time = 0.0;
    auto count = 0;
    cv::namedWindow("corners", WINDOW_NORMAL);
    Mat frame;
    cv::Size size(1920, 1080);
    Detector detector(size);
    while (capture.read(frame))
    {
        auto total = tic();
        auto corners = detector.process(frame);
        avg_time += toc(total, "total");
        ++count;

        detector.showResult("corners", corners, frame);

        auto key = waitKey(1);
        if (key == 'q' || key == 'Q')
            break;
    }

    capture.release();

    std::cout << "***************** Average Time: " << avg_time / count << "s *****************" << std::endl;
}

void test_video_2()
{
    VideoCapture capture_left, capture_right;
    capture_left.open("../../video20200120/WIN_20200120_11_37_50_Pro.mp4");
    capture_right.open("../../video20200120/WIN_20200120_11_39_46_Pro.mp4");
    if (!capture_left.isOpened() || !capture_right.isOpened())
    {
        printf("can not open ...\n");
        return;
    }

    cv::Size size(1920, 1080);
    auto detector_left = std::make_unique<Detector>(size);
    auto detector_right = std::make_unique<Detector>(size);

    cv::namedWindow("left", WINDOW_NORMAL);
    cv::namedWindow("right", WINDOW_NORMAL);

    Mat frame_left, frame_right;

    auto avg_time = 0.0;
    auto count = 0;
    auto total = tic();
    while (true)
    {
        if (!capture_left.read(frame_left) || !capture_right.read(frame_right))
            break;

        auto tt = tic();
#ifdef USE_MULTI_THREAD
        auto fut_left = std::async(std::launch::async, [&]() { return detector_left->process(frame_left); });
        auto fut_right = std::async(std::launch::async, [&]() { return detector_right->process(frame_right); });
        auto corners_left = fut_left.get();
        auto corners_right = fut_right.get();
#else
        auto corners_left = detector_left->process(frame_left);
        auto corners_right = detector_right->process(frame_right);
#endif // USE_MULTI_THREAD
        avg_time += toc(tt, "tt");
        ++count;

        detector_left->showResult("left", corners_left, frame_left);
        detector_right->showResult("right", corners_right, frame_right);

        auto key = waitKey(1);
        if (key == 'q' || key == 'Q')
            break;
    }
    auto total_ = toc(total, "total");

    capture_left.release();
    capture_right.release();

    std::cout << "***************** Total Average Time: " << total_ / count << "s *****************" << std::endl;
    std::cout << "***************** Algorithm Average Time: " << avg_time / count << "ms *****************" << std::endl;
}

int main()
{
    ////////////////
    //cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
    ////////////////

    test_video_2();
    waitKey(0);
    return 0;
}