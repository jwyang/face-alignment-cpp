#ifndef IO_LANDMARK_H
#define IO_LANDMARK_H

#include <opencv2/core/core.hpp>


#include "base.h"
#include "params.h"

class cIBug:public cLandMark
{
public:	
	cIBug(sParams *params);	
	~cIBug();
	void GetMaterial();	
	sTrainData& GetTrainData() { return m_TrainData; }
	
private:
	void  __load();
	void  __flip();
	void  __parase(const std::vector<std::string>& paths);
	void  __clip(cv::Rect& box, cv::Rect& roi, cv::Mat_<double> &land, int width, int height, double scale);
	void  __init();
	
	cv::Rect  __enlargebox(const cv::Rect &rect);
	void __flipshape(cv::Mat_<double> &land, int width);
	void __exchange(cv::Mat_<double> &land, int index1, int index2);
private:		
	sParams *m_Params;
	sTrainData m_TrainData;	
	int  m_Pictures;	
	int* m_Identity;
};
#endif