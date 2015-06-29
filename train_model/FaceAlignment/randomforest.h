#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H
#include <windows.h>
#include "params.h"
#include "base.h"
#include <opencv2/highgui/highgui.hpp>
#include <boost/random.hpp>

typedef struct sRF{
	cv::Mat_<int>*    __ind_samples;
	cv::Mat_<int>     __issplit;
	cv::Mat_<int>     __pnode;
	cv::Mat_<int>     __depth;
	cv::Mat_<int>     __cnodes;
	cv::Mat_<int>     __isleafnode;
	cv::Mat_<double>  __feat;
	cv::Mat_<double>  __thresh;
	cv::Mat_<int>     __id_leafnodes;
	int               __num_leafnodes;
	int               __num_nodes;	
}sRF;

typedef struct sRandomForest{
	sRF**  __RF;
	int    __rows;
	int    __cols;
}sRandomForest;

typedef struct sRT{
	double            __thresh;
	cv::Mat_<double>  __feat;
	cv::Mat_<int>     __lcind;
	cv::Mat_<int>     __rcind;
}sRT;


class cRandomForest{
public:
	cRandomForest(sTrainData  *data,sParams *params);
	~cRandomForest();
	void Train_Random_Forest(sRandomForest* random_forest, int stage = 0);
	void ReleaseRF(sRandomForest* rf);
private:
	void __init();
	void __splitnode(int lmarkID, int stage, int data_t,const cv::Mat_<int> &ind_samples, sRT* rt);
	void __getproposals(int num_proposals, cv::Mat_<double>& radiuspairs, cv::Mat_<double>& anglepairs);
	cv::Mat_<int> __randperm(int permutation);
	double __rand_01();	

private:	
	cv::Mat_<int>         *m_Data;
	sTrainData            *m_TrainData;
	double                m_Q;
	sParams               *m_Params;
};
#endif