#include "detector.h"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <iomanip>
#include <array>
#include "../alg/math_helper.h"
#include "../alg/timer.h"

#ifdef USE_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#endif

namespace
{
	PixelType& imgAt(cv::Mat& img, int x, int y)
	{
		return img.ptr<PixelType>(y)[x];
	}
}

Detector::Detector(const cv::Size& size)
	: SIGMA(2)
	, HALF_PATCH_SIZE(3)
	, PATCH_X(calcPatchX())
	, WIDTH_MIN(10)
	, CORR_THRESHOLD(0.7f)
	, rect{ cv::Range(0, size.width), cv::Range(0, size.height) }
{
#ifdef USE_CUDA
	initCuda(size);
#endif
}

Corners Detector::process(const cv::Mat& image)
{
	auto image_roi = image(rect.range_y, rect.range_x).clone();

	//cv::imshow("image_roi", image_roi);
	auto [corners, is_vaild] = detectCorners(image_roi);

	if (is_vaild)
	{
		for (auto& corner : corners)
			corner.point += Corner(rect.range_x.start, rect.range_y.start);

		PixelType width_sum = 0;
		Corner point_sum(0, 0);
		for (const auto& corner : corners)
		{
			point_sum += corner.point;
			width_sum += corner.width;
		}
		auto px_avg = point_sum.x / corners.size();
		auto py_avg = point_sum.y / corners.size();
		auto width_avg = width_sum / corners.size();

		auto rect_side = round(width_avg * 20);
		rect.range_x = cv::Range(
			std::max(int(px_avg - rect_side / 2), 0),
			std::min(int(px_avg + rect_side / 2), image.cols));
		rect.range_y = cv::Range(
			std::max(int(py_avg - rect_side / 2), 0),
			std::min(int(py_avg + rect_side / 2), image.rows));
	}

	Corners res;
	for (auto&& p : corners)
		res.emplace_back(p.point);

	return res;
}

std::tuple<CornersTemplate, bool> Detector::detectCorners(const cv::Mat& image)
{
	auto t0 = tic();
	gray_image = convertToGray(image);
	toc(t0, "t0:");

	auto t1 = tic();
#ifdef USE_CUDA
	std::tie(I_angle, I_weight, cmax) = secondDerivCornerMetricCuda();
#else
	std::tie(I_angle, I_weight, cmax) = secondDerivCornerMetric();
#endif
	toc(t1, "t1:");

	auto t2 = tic();
	auto corners = nonMaximumSuppression(cmax, WIDTH_MIN / 2, WIDTH_MIN);
	std::sort(corners.begin(), corners.end(),
		[](const auto& lhs, const auto& rhs) { return lhs.val > rhs.val; });
	toc(t2, "t2:");

	auto t3 = tic();
	auto [corners_on_marker, is_vaild] = detectCornersOnMarker(corners);
	toc(t3, "t3:");

	return { corners_on_marker, is_vaild };
}

void Detector::showResult(const Corners& corners, const cv::Mat& image)
{
	bool is_first = true;
	for (const auto& sc : corners)
	{
		if (is_first)
		{
			cv::circle(image, sc, 3, cv::Scalar(0, 255, 0), -1);
			is_first = !is_first;
		}
		else
		{
			cv::circle(image, sc, 3, cv::Scalar(0, 0, 255), -1);
		}
		//cv::putText(image, std::to_string(sc.score), sc.corner.point, cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 255));
	}
	cv::imshow("corners", image);
	// std::cout << "corners size: " << corners.size() << std::endl;
}

void Detector::dump(const cv::String& name, const cv::Mat& mat)
{
	auto width = mat.cols;
	auto height = mat.rows;

	std::cout.setf(std::ios::fixed);
	std::cout << name << std::endl;
	for (int i = 0; i < std::min(width, 10); ++i)
	{
		for (int j = 0; j < std::min(height, 6); ++j)
		{
			std::cout << mat.ptr<PixelType>(i)[j] << std::setprecision(6) << '\t';
		}
		std::cout << std::endl;
	}
}

cv::Mat Detector::convertToGray(const cv::Mat& image)
{
	cv::Mat gray;
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	gray.convertTo(gray, MatType);
	gray = gray / 255;

	return gray;
}

std::tuple<cv::Mat, cv::Mat, cv::Mat> Detector::secondDerivCornerMetric()
{
	cv::Mat gaussian_image;
	cv::GaussianBlur(gray_image, gaussian_image, cv::Size(7 * SIGMA + 1, 7 * SIGMA + 1), SIGMA);

	cv::Mat dx = (cv::Mat_<PixelType>(1, 3) << -1, 0, 1);
	cv::Mat dy;
	cv::transpose(dx, dy);

	// first derivative
	auto Ix = conv2(gaussian_image, dx, "same");
	auto Iy = conv2(gaussian_image, dy, "same");
	auto I_45 = Ix * cos(PI / 4) + Iy * sin(PI / 4);
	auto I_n45 = Ix * cos(-PI / 4) + Iy * sin(-PI / 4);

	// second derivative
	auto Ixy = conv2(Ix, dy, "same");
	auto I_45_x = conv2(I_45, dx, "same");
	auto I_45_y = conv2(I_45, dy, "same");
	auto I_45_45 = I_45_x * cos(-PI / 4) + I_45_y * sin(-PI / 4);

	auto cxy = static_cast<cv::Mat>(pow(SIGMA, 2) * cv::abs(Ixy) - 1.5 * SIGMA * (cv::abs(I_45) + cv::abs(I_n45)));
	auto c45 = static_cast<cv::Mat>(pow(SIGMA, 2) * cv::abs(I_45_45) - 1.5 * SIGMA * (cv::abs(Ix) + cv::abs(Iy)));
	auto cmax = static_cast<cv::Mat>(cv::max(cxy, c45));
	cv::Mat zeros_mat = cv::Mat::zeros(cmax.size(), MatType);
	cmax = cv::max(cmax, zeros_mat);

	cv::Mat I_angle, I_weight;
	cv::phase(Ix, Iy, I_angle);
	cv::magnitude(Ix, Iy, I_weight);

	return { I_angle, I_weight, cmax };
}

Maximas Detector::nonMaximumSuppression(const cv::Mat& img, int n, int margin, PixelType tau)
{
	auto width = img.cols;
	auto height = img.rows;

	Maximas maxima;
	for (int i = n + margin; i < width - n - margin; i += n + 1)
	{
		for (int j = n + margin; j < height - n - margin; j += n + 1)
		{
			auto max_i = i;
			auto max_j = j;
			auto max_val = img.ptr<PixelType>(j)[i];

			for (int i2 = i; i2 <= i + n; ++i2)
			{
				for (int j2 = j; j2 <= j + n; ++j2)
				{
					auto curr_val = img.ptr<PixelType>(j2)[i2];
					if (curr_val > max_val)
					{
						max_i = i2;
						max_j = j2;
						max_val = curr_val;
					}
				}
			}

			if (max_val < tau)
				continue;

			bool failed = false;
			for (int i2 = max_i - n;
				i2 <= std::min(max_i + n, width - margin - 1);
				i2++)
			{
				for (int j2 = max_j - n;
					j2 <= std::min(max_j + n, height - margin - 1);
					j2++)
				{
					if (img.ptr<PixelType>(j2)[i2] > max_val &&
						(i2 < i || i2 > i + n || j2 < j || j2 > j + n))
					{
						failed = true;
						break;
					}
				}
				if (failed)
					break;
			}

			if (failed)
				continue;

			maxima.emplace_back(max_i, max_j, max_val);
		}
	}

	return maxima;
}

Eigen::MatrixXf Detector::calcPatchX()
{
	std::vector<int> vec;
	for (int i = -HALF_PATCH_SIZE; i <= HALF_PATCH_SIZE; ++i)
		vec.push_back(i);

	auto size = 2 * HALF_PATCH_SIZE + 1;
	Eigen::MatrixXf XX = Eigen::MatrixXf(size * size, 6);
	for (int i = 0; i < size * size; ++i)
	{
		auto x = vec.at(i / size);
		auto y = vec.at(i % size);
		XX(i, 0) = x * x;
		XX(i, 1) = y * y;
		XX(i, 2) = x;
		XX(i, 3) = y;
		XX(i, 4) = x * y;
		XX(i, 5) = 1;
	}

	return (XX.transpose() * XX).inverse() * XX.transpose();
}

std::tuple<CornersTemplate, bool> Detector::detectCornersOnMarker(const Maximas& corners)
{
	CornersTemplate corners_selected;
	for (const auto& p : corners)
	{
		auto [corner_first, corner_second, dir] = findFirstSecondCorners(p.corner);
		if (dir != 0)
		{
			corners_selected.push_back(corner_first);
			corners_selected.push_back(corner_second);

			std::array<std::pair<int, CornerTemplate>, 2> comps = {
				std::make_pair(dir, corner_second),
				std::make_pair(-dir, corner_first) };
			for (auto& comp : comps)
			{
				while (true)
				{
					auto corner_next = predictNextCorner(comp.second, comp.first);
					if (corner_next.corr <= CORR_THRESHOLD)
						break;

					comp.second = corner_next;
					corners_selected.push_back(corner_next);
				}
			}
		}
		if (corners_selected.size() > 4)
		{
			return { corners_selected, true };
		}
	}

	return { corners_selected, false };
}

std::tuple<CornerTemplate, CornerTemplate, int> Detector::findFirstSecondCorners(const cv::Point& point)
{
	CornerTemplate corner_first(subPixelLocation(point), WIDTH_MIN);
	auto corner_second = corner_first;

	auto [angle1, angle2] = findEdgeAngles(corner_first.point);
	if (abs(angle1) < 1e-7 && abs(angle2) < 1e-7)
		return { corner_first, corner_second, 0 };

	auto template_angle = (angle1 + angle2 - PI / 2) / 2;
	auto corr = calcBolicCorrelation(corner_first.point, WIDTH_MIN, template_angle);
	if (corr <= CORR_THRESHOLD)
		return { corner_first, corner_second, 0 };

	corner_first.corr = corr;

	// optimize width and angle
	const auto DOUBLE_WIDTH_MIN = 2 * WIDTH_MIN;
	const auto CORR_THRESHOLD_EXT = 0.6f;
	auto corr_x = calcBolicCorrelation(point, DOUBLE_WIDTH_MIN, template_angle);
	auto init_width = corr_x > CORR_THRESHOLD_EXT ? DOUBLE_WIDTH_MIN : WIDTH_MIN;

	auto edge_angle1 = template_angle, edge_angle2 = template_angle + PI / 2;
	if (edge_angle2 >= PI) edge_angle2 -= PI;
	/* first--angle, second--direction */
	std::array<std::pair<PixelType, int>, 4> comps = {
		std::make_pair(edge_angle1, -1),
		std::make_pair(edge_angle1, 1),
		std::make_pair(edge_angle2, -1),
		std::make_pair(edge_angle2, 1) };

	const auto k_test = 0.8;
	const auto WIDTH_MAX = 60;
	for (const auto& comp : comps)
	{
		corner_first.width = init_width;
		corner_first.angle = comp.first;
		int dir = comp.second;

		while (corner_first.width < WIDTH_MAX)
		{
			auto next_corner = findNextCorner(corner_first, dir);

			auto width_temp = cv::norm(next_corner - corner_first.point);
			if (width_temp <= WIDTH_MIN)
			{
				corner_first.width *= 2;
				continue;
			}

			auto [angle1, angle2] = findEdgeAngles(next_corner);
			auto angle_next = abs(angle1 - corner_first.angle) < abs(angle2 - corner_first.angle) ? angle1 : angle2;
			auto corr_test_next = calcBolicCorrelation(next_corner,
				std::max(decltype(width_temp)(WIDTH_MIN), width_temp - WIDTH_MIN), angle_next);
			if (corr_test_next <= CORR_THRESHOLD)
			{
				corner_first.width *= 2;
				continue;
			}

			auto corr_test_x = calcBolicCorrelation(corner_first.point,
				std::max(decltype(width_temp)(WIDTH_MIN), width_temp * k_test), corner_first.angle);
			if (corr_test_x <= CORR_THRESHOLD_EXT)
			{
				corner_first.width *= 2;
				continue;
			}

			corner_first.width = width_temp;
			corner_second.point = subPixelLocation(next_corner);
			corner_second.angle = angle_next;
			corner_second.width = width_temp;
			corner_second.corr = corr_test_next;

			return { corner_first, corner_second, dir };
		}
	}

	return { corner_first, corner_second, 0 };
}

Corner Detector::subPixelLocation(const cv::Point& point)
{
	if (point.x < HALF_PATCH_SIZE ||
		point.y < HALF_PATCH_SIZE ||
		point.x > cmax.cols - HALF_PATCH_SIZE - 1 ||
		point.y > cmax.rows - HALF_PATCH_SIZE - 1)
	{
		return Corner(point.x, point.y);
	}

	auto width = cmax.cols, height = cmax.rows;
	auto patch = cmax(
		cv::Range(std::max(point.y - HALF_PATCH_SIZE, 0), std::min(point.y + HALF_PATCH_SIZE + 1, height)),
		cv::Range(std::max(point.x - HALF_PATCH_SIZE, 0), std::min(point.x + HALF_PATCH_SIZE + 1, width)));
	Eigen::MatrixXf e_patch;
	cv::cv2eigen(patch, e_patch);
	Eigen::Map<Eigen::RowVectorXf> v_patch(e_patch.data(), e_patch.size());
	auto beta = PATCH_X * v_patch.transpose();
	auto A = beta(0), B = beta(1), C = beta(2), D = beta(3), E = beta(4);
	auto delta = 4 * A * B - E * E;
	if (abs(delta) < 1e-7)
		return Corner(point.x, point.y);

	auto x = -(2 * B * C - D * E) / delta;
	auto y = -(2 * A * D - C * E) / delta;
	if (abs(x) > HALF_PATCH_SIZE || abs(y) > HALF_PATCH_SIZE)
		return Corner(point.x, point.y);

	return Corner(point.x + x, point.y + y);
}

std::tuple<PixelType, PixelType> Detector::findEdgeAngles(const Corner& point)
{
	auto r = 10;
	auto width = I_angle.cols, height = I_angle.rows;

	int cu = round(point.x), cv = round(point.y);
	auto u_range = cv::Range(std::max(cv - r, 0), std::min(cv + r, height));
	auto v_range = cv::Range(std::max(cu - r, 0), std::min(cu + r, width));

	return edgeOrientation(I_angle(u_range, v_range), I_weight(u_range, v_range));
}

std::tuple<PixelType, PixelType> Detector::edgeOrientation(const cv::Mat& img_angle, const cv::Mat& img_weight)
{
	const auto BIN_NUM = 32;
	using Histogram = std::array<PixelType, BIN_NUM>;
	Histogram angle_hist = {};

	/* pair: first--index, second--hist_smoothed(index) */
	using Mode = std::vector<std::pair<int, PixelType>>;

	for (int u = 0; u < img_angle.cols; ++u)
	{
		for (int v = 0; v < img_angle.rows; ++v)
		{
			auto val = [](PixelType angle) {
				angle += PI / 2;
				while (angle > PI)
					angle -= PI;
				return angle;
			}(img_angle.ptr<PixelType>(v)[u]);

			auto bin = std::max(
				std::min(
					static_cast<int>(floor(val / PI * BIN_NUM)),
					BIN_NUM - 1),
				0);

			angle_hist.at(bin) += img_weight.ptr<PixelType>(v)[u];
		}
	}

	auto findModesMeanShift = [&angle_hist, &BIN_NUM](int sigma)
	{
		Histogram hist_smoothed = {};
		Mode modes;

		for (int i = 0; i < BIN_NUM; ++i)
		{
			for (int j = -2 * sigma; j <= 2 * sigma; ++j)
			{
				auto id = (i + j + BIN_NUM) % BIN_NUM;
				hist_smoothed.at(i) += angle_hist.at(id) * normpdf(j, 0, sigma);
			}
		}

		auto is_all_zeros = [&hist_smoothed]() {
			for (const auto& hist : hist_smoothed)
				if (abs(hist - hist_smoothed.front()) >= 1e-5)
					return false;

			return true;
		};
		if (is_all_zeros())
			return modes;

		for (int i = 0; i < BIN_NUM; ++i)
		{
			auto j = i;
			while (true)
			{
				auto h0 = hist_smoothed.at(j);
				auto j1 = (j + 1 + BIN_NUM) % BIN_NUM;
				auto j2 = (j - 1 + BIN_NUM) % BIN_NUM;
				auto h1 = hist_smoothed.at(j1);
				auto h2 = hist_smoothed.at(j2);

				if (h1 >= h0 && h1 >= h2)
					j = j1;
				else if (h2 > h0&& h2 > h1)
					j = j2;
				else
					break;
			}

			auto contains = [&modes](int j) {
				for (const auto& e : modes)
					if (e.first == j)
						return true;

				return false;
			};
			if (modes.empty() || !contains(j))
			{
				modes.emplace_back(std::make_pair(j, hist_smoothed.at(j)));
			}
		}

		std::sort(modes.begin(), modes.end(),
			[](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

		return modes;
	};
	auto modes = findModesMeanShift(1);

	if (modes.size() <= 1)
		return { 0, 0 };

	PixelType angle1 = modes.at(0).first * PI / BIN_NUM;
	PixelType angle2 = modes.at(1).first * PI / BIN_NUM;
	if (angle1 > angle2)
		std::swap(angle1, angle2);

	auto delta_angle = std::min(angle2 - angle1, angle2 - angle1 + PI);
	if (delta_angle <= 0.5f)
		return { 0, 0 };

	return { angle1, angle2 };
}

PixelType Detector::calcBolicCorrelation(const Corner& point, int width, PixelType theta)
{
	auto rect_rate = 1;
	auto phi = -theta;

	auto fun_hyperbolic_tangent_scaled =
		[&theta, &phi](PixelType dx, PixelType dy, PixelType alpha = 1, PixelType beta = 1)
	{
		auto fun_hyperbolic_tangent = [&]()
		{
			auto u = -dx * sin(theta) + dy * cos(theta);
			auto v = dx * cos(phi) - dy * sin(phi);
			return tanh(alpha * u) * tanh(beta * v);
		};
		return (fun_hyperbolic_tangent() + 1) / 2; // convert to range(0, 1)
	};

	double raw_sum = 0, bolic_sum = 0;
	auto col_half_width = width, row_half_height = width * rect_rate;
	auto count = 0;
	Eigen::Matrix2f rot; rot << cos(theta), -sin(theta), sin(theta), cos(theta);
	for (int x = -col_half_width; x <= col_half_width; ++x)
	{
		for (int y = -row_half_height; y <= row_half_height; ++y)
		{
			Eigen::Vector2f d(x, y);
			Eigen::Vector2f delta = rot * d;
			int input_x = round(point.x + delta.x()), input_y = round(point.y + delta.y());

			if (input_x<0 || input_x>gray_image.cols ||
				input_y<0 || input_y>gray_image.rows)
			{
				return 0;
			}

			raw_sum += imgAt(gray_image, input_x, input_y);
			bolic_sum += fun_hyperbolic_tangent_scaled(delta.x(), delta.y());
			++count;
		}
	}
	auto raw_avg = raw_sum / count;
	auto bolic_avg = bolic_sum / count;

	double cov = 0, var_bolic = 0, var_raw = 0;
	for (int x = -col_half_width; x <= col_half_width; ++x)
	{
		for (int y = -row_half_height; y <= row_half_height; ++y)
		{
			Eigen::Vector2f d(x, y);
			Eigen::Vector2f delta = rot * d;
			int input_x = round(point.x + delta.x()), input_y = round(point.y + delta.y());

			auto diff_raw = imgAt(gray_image, input_x, input_y) - raw_avg;
			auto diff_bolic = fun_hyperbolic_tangent_scaled(delta.x(), delta.y()) - bolic_avg;

			cov += diff_raw * diff_bolic;
			var_raw += pow(diff_raw, 2);
			var_bolic += pow(diff_bolic, 2);
		}
	}

	return abs(cov / (sqrt(var_raw) * sqrt(var_bolic)));
}

Corner Detector::findNextCorner(const CornerTemplate& current, int dir)
{
	auto width = cmax.cols, height = cmax.rows;

	auto predict_x = std::min((int)abs(round(current.point.x + dir * current.width * cos(current.angle))), width);
	auto predict_y = std::min((int)abs(round(current.point.y + dir * current.width * sin(current.angle))), height);

	auto side = (int)round(std::max(current.width / 3.0, WIDTH_MIN / 2.0));

	auto cmax_sub = cmax(
		cv::Range(std::max(predict_y - side, 0), std::min(predict_y + side, height)),
		cv::Range(std::max(predict_x - side, 0), std::min(predict_x + side, width)));

	cv::Point max_pos;
	cv::minMaxLoc(cmax_sub, nullptr, nullptr, nullptr, &max_pos);

	return Corner(max_pos.x + std::max(predict_x - side, 0), max_pos.y + std::max(predict_y - side, 0));
}

CornerTemplate Detector::predictNextCorner(const CornerTemplate& current, int dir)
{
	CornerTemplate corner_next(subPixelLocation(findNextCorner(current, dir)), WIDTH_MIN);
	auto [angle1, angle2] = findEdgeAngles(corner_next.point);
	auto angle_next = abs(angle1 - current.angle) < abs(angle2 - current.angle) ? angle1 : angle2;
	auto corr_next = calcBolicCorrelation(corner_next.point,
		std::max(decltype(current.width)(WIDTH_MIN), current.width - WIDTH_MIN), angle_next);
	if (corr_next > CORR_THRESHOLD)
	{
		corner_next.corr = corr_next;
		corner_next.angle = angle_next;
		corner_next.width = cv::norm(corner_next.point - current.point);
	}

	return corner_next;
}

/////////////////////////////////////CUDA/////////////////////////////////////
#ifdef USE_CUDA
void Detector::initCuda(const cv::Size& size)
{
	/* cuda first initialization */
	auto A = cv::Mat::ones(cv::Size(3, 3), MatType);
	auto B = cv::Mat::ones(cv::Size(2, 2), MatType);
	cv::cuda::GpuMat gA(A), gR;
	cv::cuda::add(gA, gA, gR);

	cv::Mat F = cudaFilter(A, B);

	/* filter initialization */
	cv::Mat dx = (cv::Mat_<PixelType>(1, 3) << -1, 0, 1);
	cv::Mat dy;
	cv::transpose(dx, dy);

	auto filter = [](const cv::Mat& kernel, cv::Ptr<cv::cuda::Filter>& f) {
		cv::Mat flip_kernel;
		cv::flip(kernel, flip_kernel, -1);
		cv::Point anchor(flip_kernel.cols - flip_kernel.cols / 2 - 1, flip_kernel.rows - flip_kernel.rows / 2 - 1);

		f = cv::cuda::createLinearFilter(MatType, MatType, flip_kernel, anchor, cv::BORDER_CONSTANT);
	};
	filter(dx, filter_dx);
	filter(dy, filter_dy);
	filter_G = cv::cuda::createGaussianFilter(MatType, MatType, cv::Size(7 * SIGMA + 1, 7 * SIGMA + 1), SIGMA);

	/* mat initialization */
	cv::Mat ones_mat = cv::Mat::ones(size, MatType);
	cv::Mat zeros_mat = cv::Mat::zeros(size, MatType);
	g_ones.upload(ones_mat);
	g_zeros.upload(zeros_mat);
}

std::tuple<cv::Mat, cv::Mat, cv::Mat> Detector::secondDerivCornerMetricCuda()
{
	auto ttt1 = tic();
	cv::cuda::GpuMat g_gray_image(gray_image), g_gaussian_image;
	filter_G->apply(g_gray_image, g_gaussian_image);
	toc(ttt1, "ttt1");

	// first derivative
	auto ttt2 = tic();
	cv::cuda::GpuMat g_Ix, g_Iy, g_I_45, g_I_n45;
	filter_dx->apply(g_gaussian_image, g_Ix);
	filter_dy->apply(g_gaussian_image, g_Iy);
	cv::cuda::GpuMat temp1, temp2;
	cv::cuda::multiply(g_Ix, g_ones, temp1, cos(PI / 4));
	cv::cuda::multiply(g_Iy, g_ones, temp2, sin(PI / 4));
	cv::cuda::add(temp1, temp2, g_I_45);
	cv::cuda::subtract(temp1, temp2, g_I_n45);
	toc(ttt2, "ttt2");

	// second derivative
	auto ttt3 = tic();
	cv::cuda::GpuMat g_Ixy, g_I_45_x, g_I_45_y, g_I_45_45;
	filter_dy->apply(g_Ix, g_Ixy);
	filter_dx->apply(g_I_45, g_I_45_x);
	filter_dy->apply(g_I_45, g_I_45_y);
	cv::cuda::multiply(g_I_45_x, g_ones, temp1, cos(-PI / 4));
	cv::cuda::multiply(g_I_45_y, g_ones, temp2, sin(-PI / 4));
	cv::cuda::add(temp1, temp2, g_I_45_45);
	toc(ttt3, "ttt3");

	// cmax
	auto ttt4 = tic();
	auto sigma_2 = pow(SIGMA, 2), sigma_n15 = -1.5 * SIGMA;
	cv::cuda::GpuMat g_cxy, g_c45, g_cmax;
	cv::cuda::abs(g_I_45, temp1);
	cv::cuda::abs(g_I_n45, temp2);
	cv::cuda::add(temp1, temp2, temp1);
	cv::cuda::multiply(temp1, g_ones, temp1, sigma_n15);
	cv::cuda::abs(g_Ixy, temp2);
	cv::cuda::scaleAdd(temp2, sigma_2, temp1, g_cxy);

	cv::cuda::abs(g_Ix, temp1);
	cv::cuda::abs(g_Iy, temp2);
	cv::cuda::add(temp1, temp2, temp1);
	cv::cuda::multiply(temp1, g_ones, temp1, sigma_n15);
	cv::cuda::abs(g_I_45_45, temp2);
	cv::cuda::scaleAdd(temp2, sigma_2, temp1, g_c45);

	cv::cuda::max(g_cxy, g_c45, g_cmax);
	cv::cuda::max(g_cmax, g_zeros, g_cmax);

	cv::cuda::GpuMat g_I_angle, g_I_weight;
	cv::cuda::phase(g_Ix, g_Iy, g_I_angle);
	cv::cuda::magnitude(g_Ix, g_Iy, g_I_weight);
	toc(ttt4, "ttt4");

	// download
	auto ttt5 = tic();
	cv::Mat I_angle, I_weight, cmax;
	g_I_angle.download(I_angle);
	g_I_weight.download(I_weight);
	g_cmax.download(cmax);
	toc(ttt5, "ttt5");

	return { I_angle, I_weight, cmax };
}
#endif
/////////////////////////////////////CUDA/////////////////////////////////////