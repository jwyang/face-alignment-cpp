#ifndef DERIVEBINARYFEAT_H
#define DERIVEBINARYFEAT_H

#include "randomforest.h"

typedef struct sBF{
	cv::Mat_<double>* __feats;
	cv::Mat_<double>* __threshs;
	cv::Mat_<int>*    __isleaf;
	cv::Mat_<int>*    __cnodes;
}sBF;

class cBinaryFeat{
public:
	cBinaryFeat(){}
	cBinaryFeat(sTrainData* data, sParams* params);
	cv::Mat_<int> DerivBinaryfeat(sRandomForest* randf, int stage = 0);
private:
	cv::Mat_<int> __lbf_fast(const sRF*rf,const cv::Mat_<double>&feat, const cv::Mat_<double>&threshs,\
		const cv::Mat_<int>& isleaf, const cv::Mat_<int>& cnode, const sData* data, int stage, int landID);
	int __getindex(const cv::Mat_<int>&src, int val);

private:
	sTrainData*    m_TrainData;
	sParams*       m_Params;
};
#endif