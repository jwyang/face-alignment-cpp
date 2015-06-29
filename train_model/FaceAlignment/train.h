#ifndef TRAIN_H
#define TRAIN_H

#include "landmark.h"
#include "randomforest.h"
#include "derivebinaryfeat.h"
#include "globalregression.h"
#include "utils.h"
#include "head.h"


typedef struct Statistics{
	double _tx;
	double _ty;
	double _sx;
	double _sy;
}Statistics;

class cTrain{
public:
	cTrain(sParams *params);
	~cTrain();
	void Train();
private:
	cv::Mat_<double> __calcmeanface();	
	void __init_train_data();
	void __rf2model(const sRandomForest *rf, cv::Mat_<double>& RF);
	void __save_model(std::string filename, cv::Mat_<float>&rf, cv::Mat_<float>&w);
	//mean face
	cv::Mat_<double>  __procrustes(const cv::Mat_<double>& groundtruth, const std::vector<cv::Rect>& faceboxes);
	cv::Mat_<double>  __rot_scale_align(const cv::Mat_<double> &src, const cv::Mat_<double> &dst);
	
	cv::Mat_<double> __align(const cv::Mat_<double>& mean,const cv::Rect& faceBox);
	Statistics cTrain::__statistics(std::vector<cv::Rect>& faceboxes, const cv::Mat_<double>& landmarks, const cv::Mat_<double>& meanface);
	double __calctranslation(const cv::Mat_<double>& landmark, const cv::Mat_<double>& initshape);
	double __calcscaleratio(const cv::Mat_<double>& landmark, const cv::Mat_<double>& initshape);
	void  __rescale(cv::Mat_<double>& mean, const Statistics & statistics);
	
private:
	cIBug *m_IBug;
	cRandomForest *m_RF;
	sRandomForest *m_RFS;
	sParams *m_Params;
	sTrainData m_TrainData;
	cBinaryFeat *m_BinayFeat;
	cRegression  *m_Regression;
	cUtils      m_Utils;
	sHead       m_Head;
};
#endif