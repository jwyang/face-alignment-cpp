#include "derivebinaryfeat.h"
#include <math.h>
#include <fstream>
#include <boost/lexical_cast.hpp>

cBinaryFeat::cBinaryFeat(sTrainData* data, sParams* params)
{
	m_TrainData = data;
	m_Params = params;
}

cv::Mat_<int> cBinaryFeat::DerivBinaryfeat(sRandomForest *randf, int stage)
{
	if (!randf || !randf->__RF){
		cv::Mat_<int> mat;
		return mat;
	}

	sRF** rf = randf->__RF;	

	cv::Mat_<double>* __feats;
	cv::Mat_<double>* __threshs;
	cv::Mat_<int>*    __isleaf;
	cv::Mat_<int>*    __cnodes;

	try{
		__feats   = new cv::Mat_<double>[m_TrainData->__numlandmarks];
		__threshs = new cv::Mat_<double>[m_TrainData->__numlandmarks];
		__isleaf  = new cv::Mat_<int>[m_TrainData->__numlandmarks];
		__cnodes  = new cv::Mat_<int>[m_TrainData->__numlandmarks];
	}catch (std::exception e){
		std::cout << e.what() << std::endl;
	}

	for (int l=0; l < m_TrainData->__numlandmarks; l++){		
		int num_rfnodes = 0;

		for (int t = 0; t < m_Params->__max_numtrees; t++){
			num_rfnodes += rf[l][t].__num_nodes;
		}
	
		int low = 0, high = 0;

		__feats[l] = cv::Mat::zeros(num_rfnodes, 4, CV_64FC1);
		__threshs[l] = cv::Mat::zeros(num_rfnodes, 1, CV_64FC1);
		__isleaf[l] = cv::Mat::zeros(num_rfnodes, 1, CV_32SC1);
		__cnodes[l] = cv::Mat::zeros(num_rfnodes, 2, CV_32SC1);

		for (int t = 0; t < m_Params->__max_numtrees; t++){
			high = low + m_Params->__max_nodes;
			rf[l][t].__feat.copyTo(__feats[l].rowRange(low,high));			
			rf[l][t].__thresh.copyTo(__threshs[l].rowRange(low, high));
			rf[l][t].__isleafnode.copyTo(__isleaf[l].rowRange(low, high));
			rf[l][t].__cnodes.copyTo(__cnodes[l].rowRange(low, high));
			low = high;
		}		

	}	

	cv::Mat_<int> binfeatures;
	 
	for (int i=0; i < m_TrainData->__dbsizes; i++){
		cv::Mat_<int>* binfeature_lmarks = new cv::Mat_<int>[m_TrainData->__numlandmarks];
		cv::Mat_<int>  num_leafnodes = cv::Mat::zeros(1, m_TrainData->__numlandmarks, CV_32SC1);
	
		for (int l =0; l < m_TrainData->__numlandmarks; l++){
			binfeature_lmarks[l] = __lbf_fast(rf[l], __feats[l], __threshs[l], __isleaf[l], __cnodes[l], &(m_TrainData->__data[i]), stage,l);			
			num_leafnodes(0, l) = binfeature_lmarks[l].cols;			
		}

		int num_cols = (int)cv::sum(num_leafnodes).val[0];
		
		cv::Mat_<int> binfeature_alllmarks = cv::Mat::zeros(1, num_cols, CV_32SC1);
		int low = 0, high = 0;

		for (int l = 0; l < m_TrainData->__numlandmarks; l++){
			high = low + binfeature_lmarks[l].cols;
			binfeature_lmarks[l].copyTo(binfeature_alllmarks.colRange(low, high));
			low = high;
		}
		binfeatures.push_back(binfeature_alllmarks);
		

		delete[] binfeature_lmarks;
	}	

	delete[] __feats;
	delete[] __threshs;
	delete[] __isleaf;
	delete[] __cnodes;

	return binfeatures;
}

cv::Mat_<int> cBinaryFeat::__lbf_fast(const sRF*rf, const cv::Mat_<double>&feat, const cv::Mat_<double>&threshs, \
	const cv::Mat_<int>& isleaf, const cv::Mat_<int>& cnode, const sData* data, int stage, int landID)
{
	if (!data || !rf){
		cv::Mat_<int> m;
		return m;
	}

	int width  = data->__width;
	int height = data->__height;	

	cv::Mat_<double> anglepairs  = feat.colRange(0, 2);
	cv::Mat_<double> radiuspairs = feat.colRange(2, 4);	

	cv::Mat_<double> pixel_a_x_imgcoord,pixel_a_y_imgcoord,pixel_b_x_imgcoord, pixel_b_y_imgcoord;

	cv::Mat_<double> angles_cos = cv::Mat::zeros(anglepairs.rows, 2, CV_64FC1);
	cv::Mat_<double> angles_sin = cv::Mat::zeros(anglepairs.rows, 2, CV_64FC1);

	for (int i = 0; i < radiuspairs.rows; i++){
		angles_cos(i, 0) = cos(anglepairs(i, 0));
		angles_cos(i, 1) = cos(anglepairs(i, 1));
		angles_sin(i, 0) = sin(anglepairs(i, 0));
		angles_sin(i, 1) = sin(anglepairs(i, 1));
	}
	     
	cv::multiply(angles_cos.col(0), radiuspairs.col(0), pixel_a_x_imgcoord);
	cv::multiply(angles_sin.col(0), radiuspairs.col(0), pixel_a_y_imgcoord);
	pixel_a_x_imgcoord *= (m_Params->__max_raio_radius[stage] * data->__intermediate_bboxes[stage].width);
	pixel_a_y_imgcoord *= (m_Params->__max_raio_radius[stage] * data->__intermediate_bboxes[stage].height);

	cv::multiply(angles_cos.col(1), radiuspairs.col(1), pixel_b_x_imgcoord);
	cv::multiply(angles_sin.col(1), radiuspairs.col(1), pixel_b_y_imgcoord);
	pixel_b_x_imgcoord *= (m_Params->__max_raio_radius[stage] * data->__intermediate_bboxes[stage].width);
	pixel_b_y_imgcoord *= (m_Params->__max_raio_radius[stage] * data->__intermediate_bboxes[stage].height);

	//pixel_a_x_imgcoord = pixel_a_x_imgcoord * data->__tf2meanshape.at<double>(0, 0) + pixel_a_y_imgcoord * data->__tf2meanshape.at<double>(0, 1) + data->__tf2meanshape.at<double>(0, 2);
	//pixel_a_y_imgcoord = pixel_a_x_imgcoord * data->__tf2meanshape.at<double>(1, 0) + pixel_a_y_imgcoord * data->__tf2meanshape.at<double>(1, 1) + data->__tf2meanshape.at<double>(1, 2);

	//pixel_b_x_imgcoord = pixel_b_x_imgcoord * data->__tf2meanshape.at<double>(0, 0) + pixel_b_y_imgcoord * data->__tf2meanshape.at<double>(0, 1) + data->__tf2meanshape.at<double>(0, 2);
	//pixel_b_y_imgcoord = pixel_b_x_imgcoord * data->__tf2meanshape.at<double>(1, 0) + pixel_b_y_imgcoord * data->__tf2meanshape.at<double>(1, 1) + data->__tf2meanshape.at<double>(1, 2);


	cv::Mat_<double> pixel_a_x, pixel_a_y, pixel_b_x, pixel_b_y;
	

	pixel_a_x = pixel_a_x_imgcoord + data->__intermediate_shapes[stage](landID,0);
	pixel_a_y = pixel_a_y_imgcoord + data->__intermediate_shapes[stage](landID,1);

	pixel_b_x = pixel_b_x_imgcoord + data->__intermediate_shapes[stage](landID, 0);
	pixel_b_y = pixel_b_y_imgcoord + data->__intermediate_shapes[stage](landID, 1);	

	cv::Mat_<double> pdfeats = cv::Mat::zeros(pixel_a_x.rows, 1, CV_64FC1);		

	for (int k = 0; k < pixel_a_x.rows; k++){
		int a_x = (int)(pixel_a_x(k, 0) + 0.5);
		int a_y = (int)(pixel_a_y(k, 0) + 0.5);
		int b_x = (int)(pixel_b_x(k, 0) + 0.5);
		int b_y = (int)(pixel_b_y(k, 0) + 0.5);

		a_x = MAX(0, MIN(a_x, width - 1));
		a_y = MAX(0, MIN(a_y, height - 1));
		b_x = MAX(0, MIN(b_x, width - 1));
		b_y = MAX(0, MIN(b_y, height - 1));

		double pixel_v_a = data->__img_gray(cv::Point(a_x, a_y));
		double pixel_v_b = data->__img_gray(cv::Point(b_x, b_y));	
		
		pdfeats(k, 0) = pixel_v_a - pixel_v_b;
	}

	cv::Mat_<int> cind = cv::Mat::zeros(pdfeats.rows, 1, CV_32SC1);

	for (int i = 0; i < cind.rows; i++){
		if (pdfeats(i, 0) >= threshs(i,0)){
			cind(i, 0) = 1;
		}else{
			cind(i, 0) = 0;
		}
	}	

	cv::Mat_<int> binfeature = cv::Mat::zeros(1,(int)cv::sum(isleaf).val[0],CV_32SC1);

	int cumnum_nodes = 0;
	int cumnum_leafnodes = 0;

	for (int t=0;  t < m_Params->__max_numtrees; t++){
		int num_nodes = rf[t].__num_nodes;
		int id_cnode = 0;
		while (1){
			if (isleaf(id_cnode + cumnum_nodes)){
				binfeature(0, cumnum_leafnodes + __getindex(rf[t].__id_leafnodes, id_cnode)) = 1;
				cumnum_nodes     = cumnum_nodes + num_nodes;
				cumnum_leafnodes = cumnum_leafnodes + rf[t].__num_leafnodes;
				break;
			}
			id_cnode = cnode(cumnum_nodes + id_cnode, cind(cumnum_nodes + id_cnode,0));
		}
	}	
	return binfeature;
}

int cBinaryFeat::__getindex(const cv::Mat_<int>&src, int val)
{
	for (int i =0; i < src.rows; i++){
		if (src(i, 0) == val){
			return i;
		}
	}
	return -1;
}