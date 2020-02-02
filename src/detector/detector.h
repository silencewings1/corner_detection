#pragma once
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include "../alg/math_helper.h"
#include <iostream>
#include <iomanip>
#include <array>


class Detector
{
public:
	Detector(const cv::Mat& image)
		: image(image)
	{
		findCorners();
	}

	void findCorners()
	{
		auto t1 = tic();
		auto [Ix, Iy, cmax] = secondDerivCornerMetric();
		toc(t1, "t1:");

		auto t2 = tic();
		cv::Mat I_angle; cv::phase(Ix, Iy, I_angle);
		cv::Mat I_weight; cv::magnitude(Ix, Iy, I_weight);
		toc(t2, "t2:");

		auto t3 = tic();
		auto corners = nonMaximumSuppression(cmax);
		toc(t3, "t3:");

		std::sort(corners.begin(), corners.end(),
			[](const auto& lhs, const auto& rhs) { return lhs.val > rhs.val; });

		auto t4 = tic();
		auto [refined_corners, angles] = refineCorners(corners, I_angle, I_weight);
		toc(t4, "t4:");

		auto t5 = tic();
		auto corners_with_angle = subPixelLocation(cmax, refined_corners, angles);
		toc(t5, "t5:");

		auto t6 = tic();
		auto scored_corners = scoreCorners(I_angle, I_weight, corners_with_angle);
		eraseLowScoreCorners(scored_corners, 0.01);
		toc(t6, "t6:");


		auto img4 = image;
		for (auto& sc : scored_corners)
		{
			cv::circle(img4, sc.corner.point, 3, cv::Scalar(0, 0, 255), -1);
			cv::putText(img4, std::to_string(sc.score), sc.corner.point, cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 255));
		}
		cv::imshow("scored_corners", img4);
		std::cout << "scored_corners size: " << scored_corners.size() << std::endl;
	}

private:
	void dump(const cv::String& name, const cv::Mat& mat)
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
	};

	std::tuple<cv::Mat, cv::Mat, cv::Mat> secondDerivCornerMetric()
	{
		auto convertToCV_32F = [&](cv::Mat& img)
		{
			img.convertTo(img, MatType);
			img = img / 255;
		};

		auto sigma = 2;
		auto gaussian_kernel_size = round(sigma * 7) + 1;
		//cv::Mat gray_image, gaussian_image;
		cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
		convertToCV_32F(gray_image);
		//cv::GaussianBlur(gray_image, gaussian_image, cv::Size(gaussian_kernel_size, gaussian_kernel_size), sigma);
		//convertToCV_32F(gaussian_image);

		cv::Mat G = (cv::Mat_<PixelType>(15, 15) <<
			1.90451441501264e-07, 9.67192226178406e-07, 3.82531946034795e-06, 1.17828134542578e-05, 2.82655000888421e-05, 5.28069062757794e-05, 7.68335952638070e-05, 8.70638696167456e-05, 7.68335952638070e-05, 5.28069062757794e-05, 2.82655000888421e-05, 1.17828134542578e-05, 3.82531946034795e-06, 9.67192226178406e-07, 1.90451441501264e-07,
			9.67192226178406e-07, 4.91180741403700e-06, 1.94265751707265e-05, 5.98380641576443e-05, 0.000143544053746591, 0.000268175598125502, 0.000390193192882707, 0.000442146812912245, 0.000390193192882707, 0.000268175598125502, 0.000143544053746591, 5.98380641576443e-05, 1.94265751707265e-05, 4.91180741403700e-06, 9.67192226178406e-07,
			3.82531946034795e-06, 1.94265751707265e-05, 7.68335952638070e-05, 0.000236664134694527, 0.000567727745686803, 0.00106065506580148, 0.00154324401461245, 0.00174872456786274, 0.00154324401461245, 0.00106065506580148, 0.000567727745686803, 0.000236664134694527, 7.68335952638070e-05, 1.94265751707265e-05, 3.82531946034795e-06,
			1.17828134542578e-05, 5.98380641576443e-05, 0.000236664134694527, 0.000728976855220689, 0.00174872456786274, 0.00326704760457197, 0.00475352621580118, 0.00538645087804772, 0.00475352621580118, 0.00326704760457197, 0.00174872456786274, 0.000728976855220689, 0.000236664134694527, 5.98380641576443e-05, 1.17828134542578e-05,
			2.82655000888421e-05, 0.000143544053746591, 0.000567727745686803, 0.00174872456786274, 0.00419497216179922, 0.00783723978282210, 0.0114031165983104, 0.0129214239335160, 0.0114031165983104, 0.00783723978282210, 0.00419497216179922, 0.00174872456786274, 0.000567727745686803, 0.000143544053746591, 2.82655000888421e-05,
			5.28069062757794e-05, 0.000268175598125502, 0.00106065506580148, 0.00326704760457197, 0.00783723978282210, 0.0146418915416844, 0.0213038264869216, 0.0241403980280593, 0.0213038264869216, 0.0146418915416844, 0.00783723978282210, 0.00326704760457197, 0.00106065506580148, 0.000268175598125502, 5.28069062757794e-05,
			7.68335952638070e-05, 0.000390193192882707, 0.00154324401461245, 0.00475352621580118, 0.0114031165983104, 0.0213038264869216, 0.0309968846369868, 0.0351240718762925, 0.0309968846369868, 0.0213038264869216, 0.0114031165983104, 0.00475352621580118, 0.00154324401461245, 0.000390193192882707, 7.68335952638070e-05,
			8.70638696167456e-05, 0.000442146812912245, 0.00174872456786274, 0.00538645087804772, 0.0129214239335160, 0.0241403980280593, 0.0351240718762925, 0.0398007877120288, 0.0351240718762925, 0.0241403980280593, 0.0129214239335160, 0.00538645087804772, 0.00174872456786274, 0.000442146812912245, 8.70638696167456e-05,
			7.68335952638070e-05, 0.000390193192882707, 0.00154324401461245, 0.00475352621580118, 0.0114031165983104, 0.0213038264869216, 0.0309968846369868, 0.0351240718762925, 0.0309968846369868, 0.0213038264869216, 0.0114031165983104, 0.00475352621580118, 0.00154324401461245, 0.000390193192882707, 7.68335952638070e-05,
			5.28069062757794e-05, 0.000268175598125502, 0.00106065506580148, 0.00326704760457197, 0.00783723978282210, 0.0146418915416844, 0.0213038264869216, 0.0241403980280593, 0.0213038264869216, 0.0146418915416844, 0.00783723978282210, 0.00326704760457197, 0.00106065506580148, 0.000268175598125502, 5.28069062757794e-05,
			2.82655000888421e-05, 0.000143544053746591, 0.000567727745686803, 0.00174872456786274, 0.00419497216179922, 0.00783723978282210, 0.0114031165983104, 0.0129214239335160, 0.0114031165983104, 0.00783723978282210, 0.00419497216179922, 0.00174872456786274, 0.000567727745686803, 0.000143544053746591, 2.82655000888421e-05,
			1.17828134542578e-05, 5.98380641576443e-05, 0.000236664134694527, 0.000728976855220689, 0.00174872456786274, 0.00326704760457197, 0.00475352621580118, 0.00538645087804772, 0.00475352621580118, 0.00326704760457197, 0.00174872456786274, 0.000728976855220689, 0.000236664134694527, 5.98380641576443e-05, 1.17828134542578e-05,
			3.82531946034795e-06, 1.94265751707265e-05, 7.68335952638070e-05, 0.000236664134694527, 0.000567727745686803, 0.00106065506580148, 0.00154324401461245, 0.00174872456786274, 0.00154324401461245, 0.00106065506580148, 0.000567727745686803, 0.000236664134694527, 7.68335952638070e-05, 1.94265751707265e-05, 3.82531946034795e-06,
			9.67192226178406e-07, 4.91180741403700e-06, 1.94265751707265e-05, 5.98380641576443e-05, 0.000143544053746591, 0.000268175598125502, 0.000390193192882707, 0.000442146812912245, 0.000390193192882707, 0.000268175598125502, 0.000143544053746591, 5.98380641576443e-05, 1.94265751707265e-05, 4.91180741403700e-06, 9.67192226178406e-07,
			1.90451441501264e-07, 9.67192226178406e-07, 3.82531946034795e-06, 1.17828134542578e-05, 2.82655000888421e-05, 5.28069062757794e-05, 7.68335952638070e-05, 8.70638696167456e-05, 7.68335952638070e-05, 5.28069062757794e-05, 2.82655000888421e-05, 1.17828134542578e-05, 3.82531946034795e-06, 9.67192226178406e-07, 1.90451441501264e-07);
		cv::Mat gaussian_image = conv2(gray_image, G, "same");

		cv::Mat dx = (cv::Mat_<PixelType>(1, 3) << -1, 0, 1);
		cv::Mat dy; cv::transpose(dx, dy);

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

		auto cxy = static_cast<cv::Mat>(pow(sigma, 2) * cv::abs(Ixy)
			- 1.5 * sigma * (cv::abs(I_45) + cv::abs(I_n45)));
		auto c45 = static_cast<cv::Mat>(pow(sigma, 2) * cv::abs(I_45_45)
			- 1.5 * sigma * (cv::abs(Ix) + cv::abs(Iy)));
		auto cmax = static_cast<cv::Mat>(cv::max(cxy, c45));
		[](cv::Mat& img)
		{
			for (int i = 0; i < img.cols; ++i)
				for (int j = 0; j < img.rows; ++j)
					if (img.ptr<PixelType>(j)[i] < 0)
						img.ptr<PixelType>(j)[i] = 0;
		}(cmax);

		return { Ix, Iy, cmax };
	}

	Maximas nonMaximumSuppression(const cv::Mat& img, int n = 8, PixelType tau = 0.06f, int margin = 8)
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
							(i2<i || i2>i + n || j2<j || j2>j + n))
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

	std::tuple<Maximas, Angles> refineCorners(const Maximas& corners, const cv::Mat& I_angle, const cv::Mat& I_weight)
	{
		auto r = 6;
		auto width = gray_image.cols, height = gray_image.rows;

		Angles angles;
		Maximas refined_corners;
		for (int i = 0; i < std::min(static_cast<int>(corners.size()), 80); ++i)
		{
			auto cu = corners.at(i).corner.x, cv = corners.at(i).corner.y;
			auto u_range = cv::Range(std::max(cv - r, 0), std::min(cv + r, height) + 1);
			auto v_range = cv::Range(std::max(cu - r, 0), std::min(cu + r, width) + 1);

			auto angle = edgeOrientation(I_angle(u_range, v_range), I_weight(u_range, v_range));
			if (angle.v1.norm() < 0.1 && angle.v2.norm() < 0.1)
				continue;

			angles.emplace_back(std::move(angle));
			refined_corners.emplace_back(corners.at(i));
		}

		return { refined_corners, angles };
	}

	Angle edgeOrientation(const cv::Mat& img_angle, const cv::Mat& img_weight)
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
				auto val = [](PixelType angle)
				{
					angle += PI / 2;
					while (angle > PI) angle -= PI;
					return angle;
				}(img_angle.ptr<PixelType>(v)[u]);

				auto bin = std::max(
					std::min(
						static_cast<int>(floor(val / PI * BIN_NUM)),
						BIN_NUM - 1
					), 0
				);

				angle_hist.at(bin) += img_weight.ptr<PixelType>(v)[u];
			}
		}

		auto findModesMeanShift = [&angle_hist, &BIN_NUM](int sigma)
			-> std::tuple<Mode, Histogram>
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

			auto is_all_zeros = [&hist_smoothed]()
			{
				for (const auto& hist : hist_smoothed)
					if (abs(hist - hist_smoothed.front()) >= 1e-5)
						return false;

				return true;
			};
			if (is_all_zeros())
				return { modes, hist_smoothed };

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

					if (h1 >= h0 && h1 >= h2) j = j1;
					else if (h2 > h0&& h2 > h1) j = j2;
					else break;
				}

				auto contains = [&modes](int j)
				{
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

			return { modes, hist_smoothed };
		};
		auto [modes, hist_smoothed] = findModesMeanShift(1);

		if (modes.size() <= 1)
			return { Orientation(0, 0), Orientation(0, 0), 0 };

		struct SelectedModes { int id; PixelType angle; };
		std::array<SelectedModes, 2> selected_mode;
		for (int i = 0; i < 2; ++i)
			selected_mode.at(i) = { modes.at(i).first, modes.at(i).first * PI / BIN_NUM };
		std::sort(selected_mode.begin(), selected_mode.end(),
			[](const auto& lhs, const auto& rhs) { return lhs.angle < rhs.angle; });

		auto delta_angle = std::min(
			selected_mode.at(1).angle - selected_mode.at(0).angle,
			selected_mode.at(1).angle - selected_mode.at(0).angle + PI);
		if (delta_angle <= 0.5f)
			return { Orientation(0, 0), Orientation(0, 0), 0 };

		return { Orientation(cos(selected_mode.at(0).angle),sin(selected_mode.at(0).angle)),
			Orientation(cos(selected_mode.at(1).angle),sin(selected_mode.at(1).angle)),
			(selected_mode.at(0).angle + selected_mode.at(1).angle) / 2 };
	}

	Corners subPixelLocation(const cv::Mat& cmax, const Maximas& corners, const Angles& angles)
	{
		Corners res;
		auto [half_patch_size, X] = patch_para();

		//for (const auto& _point : corners)
		for (int id = 0; id < corners.size(); ++id)
		{
			auto& point = corners.at(id).corner;
			auto subPixelLocationImpl = [&cmax, &point](int half_patch_size, const Eigen::MatrixXf& X)
			{
				if (point.x < half_patch_size ||
					point.y < half_patch_size ||
					point.x > cmax.cols - half_patch_size - 1 ||
					point.y > cmax.rows - half_patch_size - 1)
				{
					return Point2p(point.x, point.y);
				}

				auto patch = cmax(
					cv::Range(point.y - half_patch_size, point.y + half_patch_size + 1),
					cv::Range(point.x - half_patch_size, point.x + half_patch_size + 1));
				Eigen::MatrixXf e_patch; cv::cv2eigen(patch, e_patch);
				Eigen::Map<Eigen::RowVectorXf> v_patch(e_patch.data(), e_patch.size());
				auto beta = X * v_patch.transpose();
				auto A = beta(0), B = beta(1), C = beta(2), D = beta(3), E = beta(4);
				auto delta = 4 * A * B - E * E;
				if (abs(delta) < 1e-7)
					return Point2p(point.x, point.y);

				auto x = -(2 * B * C - D * E) / delta;
				auto y = -(2 * A * D - C * E) / delta;
				if (abs(x) > half_patch_size || abs(y) > half_patch_size)
					return Point2p(point.x, point.y);

				return Point2p(point.x + x, point.y + y);
			};
			res.emplace_back(subPixelLocationImpl(half_patch_size, X), angles.at(id));
		}

		return res;
	}

	std::tuple<int, Eigen::MatrixXf> patch_para()
	{
		auto half_patch_size = 3;
		auto size = 2 * half_patch_size + 1;
		Eigen::MatrixXf X = Eigen::MatrixXf(size - 1, size * size);
		X << 0.00850340136054421, 0.00850340136054421, 0.00850340136054421, 0.00850340136054421, 0.00850340136054421, 0.00850340136054421, 0.00850340136054421, 1.41480844882132e-18, 7.32670275691235e-19, 1.01050025999997e-19, -5.30031796884888e-19, 1.01050025999997e-19, 7.32670275691235e-19, 1.41480844882132e-18, -0.00510204081632653, -0.00510204081632653, -0.00510204081632653, -0.00510204081632653, -0.00510204081632653, -0.00510204081632653, -0.00510204081632653, -0.00680272108843537, -0.00680272108843537, -0.00680272108843537, -0.00680272108843537, -0.00680272108843537, -0.00680272108843537, -0.00680272108843537, -0.00510204081632653, -0.00510204081632653, -0.00510204081632653, -0.00510204081632653, -0.00510204081632653, -0.00510204081632653, -0.00510204081632653, 1.41480844882132e-18, 7.32670275691235e-19, 1.01050025999997e-19, -5.30031796884888e-19, 1.01050025999997e-19, 7.32670275691235e-19, 1.41480844882132e-18, 0.00850340136054421, 0.00850340136054421, 0.00850340136054421, 0.00850340136054421, 0.00850340136054421, 0.00850340136054421, 0.00850340136054421,
			0.00850340136054422, 6.84818550783265e-19, -0.00510204081632653, -0.00680272108843538, -0.00510204081632653, 6.84818550783265e-19, 0.00850340136054422, 0.00850340136054422, 6.84818550783265e-19, -0.00510204081632653, -0.00680272108843538, -0.00510204081632653, 6.84818550783265e-19, 0.00850340136054422, 0.00850340136054422, 1.36963710156653e-18, -0.00510204081632653, -0.00680272108843538, -0.00510204081632653, 1.36963710156653e-18, 0.00850340136054422, 0.00850340136054422, 0, -0.00510204081632653, -0.00680272108843537, -0.00510204081632653, 0, 0.00850340136054422, 0.00850340136054422, 1.36963710156653e-18, -0.00510204081632653, -0.00680272108843538, -0.00510204081632653, 1.36963710156653e-18, 0.00850340136054422, 0.00850340136054422, 6.84818550783265e-19, -0.00510204081632653, -0.00680272108843538, -0.00510204081632653, 6.84818550783265e-19, 0.00850340136054422, 0.00850340136054422, 6.84818550783265e-19, -0.00510204081632653, -0.00680272108843538, -0.00510204081632653, 6.84818550783265e-19, 0.00850340136054422,
			-0.0153061224489796, -0.0153061224489796, -0.0153061224489796, -0.0153061224489796, -0.0153061224489796, -0.0153061224489796, -0.0153061224489796, -0.0102040816326531, -0.0102040816326531, -0.0102040816326531, -0.0102040816326531, -0.0102040816326531, -0.0102040816326531, -0.0102040816326531, -0.00510204081632653, -0.00510204081632653, -0.00510204081632653, -0.00510204081632653, -0.00510204081632653, -0.00510204081632653, -0.00510204081632653, 0, 0, 0, 0, 0, 0, 0, 0.00510204081632653, 0.00510204081632653, 0.00510204081632653, 0.00510204081632653, 0.00510204081632653, 0.00510204081632653, 0.00510204081632653, 0.0102040816326531, 0.0102040816326531, 0.0102040816326531, 0.0102040816326531, 0.0102040816326531, 0.0102040816326531, 0.0102040816326531, 0.0153061224489796, 0.0153061224489796, 0.0153061224489796, 0.0153061224489796, 0.0153061224489796, 0.0153061224489796, 0.0153061224489796,
			-0.0153061224489796, -0.0102040816326531, -0.00510204081632653, 0, 0.00510204081632653, 0.0102040816326531, 0.0153061224489796, -0.0153061224489796, -0.0102040816326531, -0.00510204081632653, 0, 0.00510204081632653, 0.0102040816326531, 0.0153061224489796, -0.0153061224489796, -0.0102040816326531, -0.00510204081632653, 0, 0.00510204081632653, 0.0102040816326531, 0.0153061224489796, -0.0153061224489796, -0.0102040816326531, -0.00510204081632653, 0, 0.00510204081632653, 0.0102040816326531, 0.0153061224489796, -0.0153061224489796, -0.0102040816326531, -0.00510204081632653, 0, 0.00510204081632653, 0.0102040816326531, 0.0153061224489796, -0.0153061224489796, -0.0102040816326531, -0.00510204081632653, 0, 0.00510204081632653, 0.0102040816326531, 0.0153061224489796, -0.0153061224489796, -0.0102040816326531, -0.00510204081632653, 0, 0.00510204081632653, 0.0102040816326531, 0.0153061224489796,
			0.0114795918367347, 0.00765306122448980, 0.00382653061224490, 0, -0.00382653061224490, -0.00765306122448980, -0.0114795918367347, 0.00765306122448980, 0.00510204081632653, 0.00255102040816327, 0, -0.00255102040816327, -0.00510204081632653, -0.00765306122448980, 0.00382653061224490, 0.00255102040816327, 0.00127551020408163, 0, -0.00127551020408163, -0.00255102040816327, -0.00382653061224490, 0, 0, 0, 0, 0, 0, 0, -0.00382653061224490, -0.00255102040816327, -0.00127551020408163, 0, 0.00127551020408163, 0.00255102040816327, 0.00382653061224490, -0.00765306122448980, -0.00510204081632653, -0.00255102040816327, 0, 0.00255102040816327, 0.00510204081632653, 0.00765306122448980, -0.0114795918367347, -0.00765306122448980, -0.00382653061224490, 0, 0.00382653061224490, 0.00765306122448980, 0.0114795918367347,
			-0.0476190476190476, -0.0136054421768707, 0.00680272108843540, 0.0136054421768708, 0.00680272108843540, -0.0136054421768707, -0.0476190476190476, -0.0136054421768708, 0.0204081632653061, 0.0408163265306123, 0.0476190476190476, 0.0408163265306123, 0.0204081632653061, -0.0136054421768708, 0.00680272108843535, 0.0408163265306122, 0.0612244897959183, 0.0680272108843537, 0.0612244897959183, 0.0408163265306122, 0.00680272108843535, 0.0136054421768707, 0.0476190476190476, 0.0680272108843537, 0.0748299319727891, 0.0680272108843537, 0.0476190476190476, 0.0136054421768707, 0.00680272108843535, 0.0408163265306122, 0.0612244897959183, 0.0680272108843537, 0.0612244897959183, 0.0408163265306122, 0.00680272108843535, -0.0136054421768708, 0.0204081632653061, 0.0408163265306123, 0.0476190476190476, 0.0408163265306123, 0.0204081632653061, -0.0136054421768708, -0.0476190476190476, -0.0136054421768707, 0.00680272108843540, 0.0136054421768708, 0.00680272108843540, -0.0136054421768707, -0.0476190476190476;

		return { half_patch_size , X };
	}

	ScoreCorners scoreCorners(const cv::Mat img_angle, const cv::Mat img_weight, const Corners& corners)
	{
		auto width = gray_image.cols, height = gray_image.rows;
		std::array<int, 3> radius = { 4, 8, 12};

		ScoreCorners scored_corners;
		for (const auto& corner : corners)
		{
			auto x = round(corner.point.x);
			auto y = round(corner.point.y);

			std::vector<PixelType> scores;
			for (const auto& r : radius)
			{
				if (x > r&& x < width - r && y > r&& y < height - r)
				{
					auto x_range = cv::Range(y - r, y + r + 1);
					auto y_range = cv::Range(x - r, x + r + 1);

					auto img_sub = gray_image(x_range, y_range).clone();
					auto img_weight_sub = img_weight(x_range, y_range).clone();
					scores.push_back(cornerCorrelationScore(img_sub, img_weight_sub, corner.angle.v1, corner.angle.v2));
				}
			}
			ScoreCorner sc = { corner, *std::max_element(scores.begin(), scores.end()) };
			scored_corners.emplace_back(sc);
		}

		return scored_corners;
	}

	PixelType cornerCorrelationScore(const cv::Mat& img, const cv::Mat& img_weight, const Orientation& v1, const Orientation& v2)
	{
		auto imgWeight = img_weight;

		//center
		int c[] = { imgWeight.cols / 2, imgWeight.cols / 2 };

		//compute gradient filter kernel(bandwith = 3 px)
		cv::Mat img_filter = cv::Mat::ones(imgWeight.size(), imgWeight.type());
		img_filter = img_filter * -1;
		for (int i = 0; i < imgWeight.cols; i++)
		{
			for (int j = 0; j < imgWeight.rows; j++)
			{
				cv::Point2f p1 = cv::Point2f(i - c[0], j - c[1]);
				cv::Point2f p2 = cv::Point2f(p1.x * v1.x() * v1.x() + p1.y * v1.x() * v1.y(),
					p1.x * v1.x() * v1.y() + p1.y * v1.y() * v1.y());
				cv::Point2f p3 = cv::Point2f(p1.x * v2.x() * v2.x() + p1.y * v2.x() * v2.y(),
					p1.x * v2.x() * v2.y() + p1.y * v2.y() * v2.y());
				float norm1 = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
				float norm2 = sqrt((p1.x - p3.x) * (p1.x - p3.x) + (p1.y - p3.y) * (p1.y - p3.y));
				if (norm1 <= 1.5 || norm2 <= 1.5)
				{
					img_filter.ptr<PixelType>(j)[i] = 1;
				}
			}
		}

		//normalize
		cv::Mat mean, std, mean1, std1;
		meanStdDev(imgWeight, mean, std);
		meanStdDev(img_filter, mean1, std1);
		for (int i = 0; i < imgWeight.cols; i++)
		{
			for (int j = 0; j < imgWeight.rows; j++)
			{
				imgWeight.ptr<PixelType>(j)[i] = (PixelType)(imgWeight.ptr<PixelType>(j)[i] - mean.ptr<double>(0)[0]) / (PixelType)std.ptr<double>(0)[0];
				img_filter.ptr<PixelType>(j)[i] = (PixelType)(img_filter.ptr<PixelType>(j)[i] - mean1.ptr<double>(0)[0]) / (PixelType)std1.ptr<double>(0)[0];
			}
		}

		//convert into vectors
		std::vector<float> vec_filter, vec_weight;
		for (int i = 0; i < imgWeight.cols; i++)
		{
			for (int j = 0; j < imgWeight.rows; j++)
			{
				vec_filter.push_back(img_filter.ptr<PixelType>(j)[i]);
				vec_weight.push_back(imgWeight.ptr<PixelType>(j)[i]);
			}
		}

		//compute gradient score
		float sum = 0;
		for (int i = 0; i < vec_weight.size(); i++)
		{
			sum += vec_weight[i] * vec_filter[i];
		}
		sum = (PixelType)sum / (PixelType)(vec_weight.size() - 1);
		PixelType score_gradient = sum >= 0 ? sum : 0;

		//create intensity filter kernel
		cv::Mat kernelA, kernelB, kernelC, kernelD;
		createkernel(atan2(v1.y(), v1.x()), atan2(v2.y(), v2.x()), c[0], kernelA, kernelB, kernelC, kernelD);//1.1 产生四种核

		//checkerboard responses
		float a1, a2, b1, b2;
		a1 = kernelA.dot(img);
		a2 = kernelB.dot(img);
		b1 = kernelC.dot(img);
		b2 = kernelD.dot(img);

		float mu = (a1 + a2 + b1 + b2) / 4;

		float score_a = (a1 - mu) >= (a2 - mu) ? (a2 - mu) : (a1 - mu);
		float score_b = (mu - b1) >= (mu - b2) ? (mu - b2) : (mu - b1);
		float score_1 = score_a >= score_b ? score_b : score_a;

		score_b = (b1 - mu) >= (b2 - mu) ? (b2 - mu) : (b1 - mu);
		score_a = (mu - a1) >= (mu - a2) ? (mu - a2) : (mu - a1);
		float score_2 = score_a >= score_b ? score_b : score_a;

		float score_intensity = score_1 >= score_2 ? score_1 : score_2;
		score_intensity = score_intensity > 0.0 ? score_intensity : 0.0;

		return score_gradient * score_intensity;
	}

	void createkernel(PixelType angle1, PixelType angle2, int kernelSize, cv::Mat& kernelA, cv::Mat& kernelB, cv::Mat& kernelC, cv::Mat& kernelD)
	{
		int width = (int)kernelSize * 2 + 1;
		int height = (int)kernelSize * 2 + 1;
		kernelA = cv::Mat::zeros(height, width, MatType);
		kernelB = cv::Mat::zeros(height, width, MatType);
		kernelC = cv::Mat::zeros(height, width, MatType);
		kernelD = cv::Mat::zeros(height, width, MatType);

		for (int u = 0; u < width; ++u) {
			for (int v = 0; v < height; ++v) {
				PixelType vec[] = { u - kernelSize, v - kernelSize };//相当于将坐标原点移动到核中心
				PixelType dis = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]);//相当于计算到中心的距离
				PixelType side1 = vec[0] * (-sin(angle1)) + vec[1] * cos(angle1);//相当于将坐标原点移动后的核进行旋转，以此产生四种核
				PixelType side2 = vec[0] * (-sin(angle2)) + vec[1] * cos(angle2);//X=X0*cos+Y0*sin;Y=Y0*cos-X0*sin
				if (side1 <= -0.1 && side2 <= -0.1) {
					kernelA.ptr<PixelType>(v)[u] = normpdf(dis, 0, kernelSize / 2);
				}
				if (side1 >= 0.1 && side2 >= 0.1) {
					kernelB.ptr<PixelType>(v)[u] = normpdf(dis, 0, kernelSize / 2);
				}
				if (side1 <= -0.1 && side2 >= 0.1) {
					kernelC.ptr<PixelType>(v)[u] = normpdf(dis, 0, kernelSize / 2);
				}
				if (side1 >= 0.1 && side2 <= -0.1) {
					kernelD.ptr<PixelType>(v)[u] = normpdf(dis, 0, kernelSize / 2);
				}
			}
		}
		//std::cout << "kernelA:" << kernelA << endl << "kernelB:" << kernelB << endl
		//	<< "kernelC:" << kernelC<< endl << "kernelD:" << kernelD << endl;
		//归一化
		kernelA = kernelA / cv::sum(kernelA)[0];
		kernelB = kernelB / cv::sum(kernelB)[0];
		kernelC = kernelC / cv::sum(kernelC)[0];
		kernelD = kernelD / cv::sum(kernelD)[0];
	}

	void eraseLowScoreCorners(ScoreCorners& scored_corners, PixelType threshold)
	{
		for (auto it = scored_corners.begin(); it != scored_corners.end();)
		{
			if (it->score < threshold) it = scored_corners.erase(it);
			else ++it;
		}
	}

private:
	const cv::Mat image;
	cv::Mat gray_image;
};

