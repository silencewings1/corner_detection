#pragma once
#include "../def/type_def.h"
#include "../def/macro_def.h"

/* full  -- Return the full convolution, including border
   same  -- Return only the part that corresponds to the original image
   valid -- Return only the submatrix containing elements that were not influenced by the border */
cv::Mat conv2(const cv::Mat& img, const cv::Mat& kernel, const cv::String& mode);
PixelType normpdf(PixelType dist, PixelType mu, PixelType sigma);

#ifdef USE_CUDA
cv::Mat cudaFilter(const cv::Mat &img, const cv::Mat &kernel); // same
#endif



