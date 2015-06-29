#ifndef GLOBALREGRESSION_H
#define GLOBALREGRESSION_H

#include "base.h"
#include "params.h"
#include "evaluate.h"
#include "utils.h"
#include "liblinear/linear.h"

class cRegression{
public:
	cRegression(sTrainData* data, sParams* params);
	cv::Mat_<double> GlobalRegression(const cv::Mat_<int>&binaryfeatures,int stage);
private:	
	cv::Mat_<double> __train_regressor(const cv::Mat_<double>& label_vec, const cv::SparseMat_<int>& instance_mat);	
	cv::Mat_<double> __train_regressor(const cv::Mat_<double>& label_vec, const cv::Mat_<int>& instance_mat);
private:
	sTrainData*      m_TrainData;
	sParams*         m_Params;	
	cUtils           m_Utils;
	cEvaluate*       m_Evaluate;
};
#endif