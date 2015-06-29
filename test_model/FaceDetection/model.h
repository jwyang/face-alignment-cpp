#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <opencv2/highgui/highgui.hpp>
#include "head.h"
#include "params.h"

typedef struct sModel{
	sHead           __head;
	cv::Mat_<float> __meanface;
	cv::Mat_<float> __rf;
	cv::Mat_<float> __w;
}sModel;

class cModel{
public:
	cModel(const std::string& model_name,sParams *params);	
	~cModel();
	cv::Mat_<int> DerivBinaryfeat( const cv::Mat_<uchar>&img, const cv::Rect& bbox, const cv::Mat_<float>& shape,int stage);
	void Init();
	void UpDate(cv::Mat_<int>&binary, cv::Rect &bbox, cv::Mat_<float>& shape, int stage);
	sModel GetModel(){ return m_Model;}
	cv::Mat_<float> GetMeanFace(){ return m_Model.__meanface; }
	cv::Mat_<float> Reshape(cv::Mat_<float>& mean, cv::Rect& faceBox);
	cv::Mat_<float> Reshape_alt(cv::Mat_<float>& mean, cv::Rect& faceBox);

private:	
	int __readmodel(sModel*model = NULL);//get meanface randfs w etc.
	cv::Mat_<int> __lbf_fast(const cv::Mat_<uchar>&img, const cv::Rect& bbox, const cv::Mat_<float>& shape, int markID, int stage);
	int __getindex(const cv::Mat_<int>&src, int val);	
private:		
	std::string  m_Name;
	cv::Mat_<float>* m_AX;
	cv::Mat_<float>* m_AY;
	cv::Mat_<float>* m_BX;
	cv::Mat_<float>* m_BY;
	cv::Mat_<float>* m_Thresh;
	cv::Mat_<int>    m_Isleaf;
	cv::Mat_<int>    m_Cnodes;
	cv::Mat_<int>    m_Idleafnodes;
	sModel  m_Model;
	sParams        *m_Params;
};

#endif