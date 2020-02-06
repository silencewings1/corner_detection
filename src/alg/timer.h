#pragma once
#include <chrono>   
#include <iostream>
#include <string>


auto tic = []()
{
	return std::chrono::system_clock::now();
};

auto toc = [](auto start, const std::string& name = "")
{
	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	auto duration_ms = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
	//std::cout << name << '\t' << duration_ms << "s" << std::endl;

	return duration_ms;
};
