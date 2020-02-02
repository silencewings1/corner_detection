#include "math_helper.h"
using namespace cv;

Mat conv2(const Mat& img, const Mat& kernel, const String& mode)
{
	if (mode != "full" && mode != "same" && mode != "valid")
	{
		printf("mode should be full, same, valid\n");
		return img;
	}

	Mat flip_kernel;
	flip(kernel, flip_kernel, -1);
	Mat source = img;
	if (mode == "full")
	{
		source = Mat();
		const int additionalRows = flip_kernel.rows - 1, additionalCols = flip_kernel.cols - 1;
		copyMakeBorder(img, source, (additionalRows + 1) / 2, additionalRows / 2, (additionalCols + 1) / 2, additionalCols / 2, BORDER_CONSTANT, Scalar(0));
	}
	Point anchor(flip_kernel.cols - flip_kernel.cols / 2 - 1, flip_kernel.rows - flip_kernel.rows / 2 - 1);
	Mat dest;
	filter2D(source, dest, img.depth(), flip_kernel, anchor, 0, BORDER_CONSTANT);

	if (mode == "valid")
	{
		dest = dest.colRange((flip_kernel.cols - 1) / 2, dest.cols - flip_kernel.cols / 2).rowRange((flip_kernel.rows - 1) / 2, dest.rows - flip_kernel.rows / 2);
	}
	return dest;
}

PixelType normpdf(PixelType dist, PixelType mu, PixelType sigma)
{
	auto s = exp(-0.5f * pow((dist - mu) / sigma, 2));
	return s / (sqrt(2 * PI) * sigma);
}


