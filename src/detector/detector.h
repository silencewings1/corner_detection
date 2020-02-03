#pragma once
#include "../def/type_def.h"
#include "../def/macro_def.h"

class Detector
{
public:
	Detector(const cv::Size &size);

	ScoreCorners findCorners(const cv::Mat &image);
	void showResult(const ScoreCorners &corners, const cv::Mat &image);

private:
	/* normal */
	cv::Mat convertToGray(const cv::Mat &image);
	std::tuple<cv::Mat, cv::Mat, cv::Mat> secondDerivCornerMetric(const cv::Mat &gray_image);
	Maximas nonMaximumSuppression(const cv::Mat &img, int n = 8, PixelType tau = 0.06f, int margin = 8);
	std::tuple<Maximas, Angles> refineCorners(const Maximas &corners, const cv::Mat &I_angle, const cv::Mat &I_weight);
	Angle edgeOrientation(const cv::Mat &img_angle, const cv::Mat &img_weight);
	Corners subPixelLocation(const cv::Mat &cmax, const Maximas &corners, const Angles &angles);
	ScoreCorners scoreCorners(const cv::Mat &gray_image, const cv::Mat img_angle, const cv::Mat img_weight, const Corners &corners);
	PixelType cornerCorrelationScore(const cv::Mat &img, const cv::Mat &img_weight, const Orientation &v1, const Orientation &v2);
	void createkernel(PixelType angle1, PixelType angle2, int kernelSize, cv::Mat &kernelA, cv::Mat &kernelB, cv::Mat &kernelC, cv::Mat &kernelD);
	void eraseLowScoreCorners(ScoreCorners &scored_corners, PixelType threshold);

#ifdef USE_CUDA
	void initCuda(const cv::Size &size);
	std::tuple<cv::Mat, cv::Mat, cv::Mat> secondDerivCornerMetricCuda(const cv::Mat &gray_image);
#endif

	/* others */
	void dump(const cv::String &name, const cv::Mat &mat);
	Eigen::MatrixXf calcPatchX();

private:
	const int sigma;
	const int half_patch_size;
	const Eigen::MatrixXf patch_X;

#ifdef USE_CUDA
	cv::Ptr<cv::cuda::Filter> filter_dx, filter_dy, filter_G;
	cv::cuda::GpuMat g_ones, g_zeros;
#endif
};
