#ifndef UTILS_H
#define UTILS_H

#include "params.h"
#include <opencv2/highgui/highgui.hpp>
class cUtils{
public:
	cUtils() {}
	cv::Mat_<double>  Calc_Point_Transform(const cv::Mat_<double>&src, const cv::Mat_<double>&dst);
	cv::Mat_<double>  Reshape(const cv::Mat& modelmean, cv::Rect& faceBox, sParams *params);
	cv::Mat_<double>  Reshape_alt(const cv::Mat& modelmean, cv::Rect& faceBox, sParams *params);
};
#endif