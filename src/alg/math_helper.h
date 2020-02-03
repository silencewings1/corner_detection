#pragma once
#include "../def/type_def.h"


/* full  -- Return the full convolution, including border
   same  -- Return only the part that corresponds to the original image
   valid -- Return only the submatrix containing elements that were not influenced by the border */
cv::Mat conv2(const cv::Mat& img, const cv::Mat& kernel, const cv::String& mode);

cv::Mat cudaConv2(const cv::Mat& img, const cv::Mat& kernel);
cv::Mat cudaFilter(const cv::Mat &img, const cv::Mat &kernel);

PixelType normpdf(PixelType dist, PixelType mu, PixelType sigma);


#include <chrono>   
#include <iostream>

auto tic = []()
{
	return std::chrono::system_clock::now();
};

auto toc = [](auto start, const cv::String& name = "")
{
	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << name << '\t' << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << "s" << std::endl;
};

