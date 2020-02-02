#pragma once
#include "../def/type_def.h"


class Optimizator
{
public:
	Optimizator(const Corners& left, const Corners& right)
		: left(left), right(right)
	{}



private:
	Corners left;
	Corners right;
};