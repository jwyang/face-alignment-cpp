#ifndef IO_BASE_H
#define IO_BASE_H

#include <string>
#include <vector>
#include <omp.h>
#include <opencv2/opencv.hpp>


typedef struct sData{

	int                            __width;
	int                            __height;
	cv::Mat_<uchar>                __img_gray;
	cv::Mat_<double>               __shape_gt;	
	cv::Rect                       __bbox_gt;
	std::vector<cv::Mat_<double>>  __intermediate_shapes;
	std::vector<cv::Rect>          __intermediate_bboxes;
	cv::Mat_<double>               __shapes_residual;
	//cv::Mat                        __tf2meanshape;
	//cv::Mat                        __meanshape2tf;
}sData;  

typedef struct sTrainData{
	int __dbsizes;
	int __numlandmarks;
	cv::Mat_<double>       __meanface;
	cv::Mat_<double>       __meanface_one_row;
	std::vector<sData>     __data;
}sTrainData;

class cLandMark{
public:
	virtual sTrainData& GetTrainData() = 0;
};

#endif