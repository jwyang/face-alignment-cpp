#ifndef EVALUATE_H
#define EVALUATE_H

#include "base.h"

class cEvaluate
{
public:
	cEvaluate();
	~cEvaluate();

public:
	double compute_error(std::vector<cv::Mat_<double>>& preshapes, std::vector<cv::Mat_<double>>& gtshapes);
};
#endif

