#include "model.h"
#include "stdio.h"
#include <iostream>
#include <fstream>

cModel::cModel(const std::string& model_name, sParams *params)
{	
	m_Name = model_name;	
	m_AX = NULL;
	m_AY = NULL;
	m_BX = NULL;
	m_BY = NULL;
	m_Thresh = NULL;
	m_Params = params;
}

cModel::~cModel()
{
	if (m_AX){
		delete[] m_AX;
	}

	if (m_AY){
		delete[] m_AY;
	}

	if (m_BX){
		delete[] m_BX;
	}

	if (m_BY){
		delete[] m_BY;
	}

	if (m_Thresh){
		delete[] m_Thresh;
	}
}

int cModel::__readmodel(sModel*model)
{
	FILE *fp = NULL;

	fp = fopen(m_Name.c_str(), "rb");
	if (fp == NULL){
		return -1;
	}

	//read head
	fread(&model->__head, sizeof(model->__head), 1, fp);

	//read meanface
	model->__meanface = cv::Mat::zeros(2, model->__head.__num_point, CV_32FC1);
	for (int i = 0; i < 2; i++){
		fread(model->__meanface.ptr<float>(i), sizeof(float)*model->__meanface.cols, 1, fp);
	}
	model->__meanface = model->__meanface.t();

	//read random forest
	model->__rf = cv::Mat::zeros(model->__head.__num_tree_total,model->__head.__dim_tree, CV_32FC1);
	for (int i = 0; i < model->__rf.rows; i++){
		fread(model->__rf.ptr<float>(i), sizeof(float)*model->__rf.cols, 1, fp);
	}	

	//read weights
	model->__w = cv::Mat::zeros(model->__head.__dim_feat* model->__head.__num_stage, model->__head.__num_point * 2, CV_32FC1);
	for (int i = 0; i < model->__w.rows; i++){
		fread(model->__w.ptr<float>(i), sizeof(float)*model->__w.cols, 1, fp);
	}	

	fclose(fp);
	return 0;
}

cv::Mat_<int> cModel::DerivBinaryfeat(const cv::Mat_<uchar>&img, const cv::Rect& bbox, const cv::Mat_<float>& shape, int stage)
{
	int cols = 0;
	int num_point = m_Model.__head.__num_point;
	int num_tree_per_point = m_Model.__head.__num_tree_per_point;
	int num_leaf = m_Model.__head.__num_leaf;
	int low = 0, high = 0;

	cv::Mat_<int> binary = cv::Mat::zeros(1, num_point * num_leaf * num_tree_per_point, CV_32SC1);
	for (int i = 0; i < num_point; i++){
		cv::Mat_<int> tmp = __lbf_fast(img, bbox, shape, i, stage);
		high = low + tmp.cols;
		tmp.copyTo(binary.colRange(low, high));
		low = high;
	}
	return binary;
}

cv::Mat_<int> cModel::__lbf_fast(const cv::Mat_<uchar>&img, const cv::Rect& bbox, const cv::Mat_<float>& shape,int markID, int stage)
{
	int max_stage = m_Model.__head.__num_stage;
	int num_node = m_Model.__head.__num_leaf + m_Model.__head.__num_node;
	int num_point = m_Model.__head.__num_point;
	int num_tree_per_point = m_Model.__head.__num_tree_per_point;
	int num_leaf = m_Model.__head.__num_leaf;
	
	m_AX[stage].row(markID) *= bbox.width;
	m_AY[stage].row(markID) *= bbox.height;
	m_BX[stage].row(markID) *= bbox.width;
	m_BY[stage].row(markID) *= bbox.height;	

	m_AX[stage].row(markID) += shape(markID, 0);
	m_AY[stage].row(markID) += shape(markID, 1);
	m_BX[stage].row(markID) += shape(markID, 0);
	m_BY[stage].row(markID) += shape(markID, 1);

	cv::Mat_<int> cind = cv::Mat::ones(m_AX[stage].cols, 1, CV_32SC1);
	cv::Mat_<float> AX = m_AX[stage].row(markID);
	cv::Mat_<float> AY = m_AY[stage].row(markID);
	cv::Mat_<float> BX = m_BX[stage].row(markID);
	cv::Mat_<float> BY = m_BY[stage].row(markID);
	cv::Mat_<float> Thresh = m_Thresh[stage].row(markID);	
	

	int width =  img.cols;
	int height = img.rows;	
	
	for (int j = 0; j < AX.cols; j += num_node){
		for (int index = 0; index < m_Model.__head.__num_node; index++){
			int pos = j + index;
			int a_x = (int)(AX(0, pos) + 0.5);
			int a_y = (int)(AY(0, pos) + 0.5);
			int b_x = (int)(BX(0, pos) + 0.5);
			int b_y = (int)(BY(0, pos) + 0.5);

			a_x = MAX(0, MIN(a_x, width - 1));
			a_y = MAX(0, MIN(a_y, height - 1));
			b_x = MAX(0, MIN(b_x, width - 1));
			b_y = MAX(0, MIN(b_y, height - 1));

			float pixel_v_a = (float)img(cv::Point(a_x, a_y));
			float pixel_v_b = (float)img(cv::Point(b_x, b_y));
			float val = pixel_v_a - pixel_v_b;
			
			if (val < (float)Thresh(0, pos)){
				cind(pos, 0) = 0;
			}
		}		
	}

	cv::Mat_<int> binfeature = cv::Mat::zeros(1, (int)cv::sum(m_Isleaf).val[0], CV_32SC1);

	int cumnum_nodes = 0;
	int cumnum_leafnodes = 0;

	for (int t = 0; t < num_tree_per_point; t++){		
		int id_cnode = 0;
		while (1){
			if (m_Isleaf(id_cnode + cumnum_nodes)){
				binfeature(0, cumnum_leafnodes + __getindex(m_Idleafnodes, id_cnode)) = 1;
				cumnum_nodes = cumnum_nodes + num_node;
				cumnum_leafnodes = cumnum_leafnodes + num_leaf;
				break;
			}
			id_cnode = m_Cnodes(cumnum_nodes + id_cnode, cind(cumnum_nodes + id_cnode, 0));
		}
	}
	return binfeature;	
}

int cModel::__getindex(const cv::Mat_<int>&src, int val)
{
	for (int i = 0; i < src.rows; i++){
		if (src(i, 0) == val){
			return i;
		}
	}
	return -1;
}

void cModel::Init()
{	
	__readmodel(&m_Model);
	int max_stage = m_Model.__head.__num_stage;
	int num_node = m_Model.__head.__num_leaf + m_Model.__head.__num_node;
	int num_point = m_Model.__head.__num_point;
	int num_tree_per_point = m_Model.__head.__num_tree_per_point;
	int node_step = m_Model.__head.__node_step;

	m_AX = new cv::Mat_<float>[max_stage];
	m_AY = new cv::Mat_<float>[max_stage];
	m_BX = new cv::Mat_<float>[max_stage];
	m_BY = new cv::Mat_<float>[max_stage];
	m_Thresh = new cv::Mat_<float>[max_stage];

	for (int i = 0; i < max_stage; i++){
		m_AX[i] = cv::Mat::zeros(num_point, num_node * num_tree_per_point, CV_32FC1);//68X630
		m_AY[i] = cv::Mat::zeros(num_point, num_node * num_tree_per_point, CV_32FC1);
		m_BX[i] = cv::Mat::zeros(num_point, num_node * num_tree_per_point, CV_32FC1);
		m_BY[i] = cv::Mat::zeros(num_point, num_node * num_tree_per_point, CV_32FC1);
		m_Thresh[i] = cv::Mat::zeros(num_point, num_node * num_tree_per_point, CV_32FC1);

		for (int j = 0; j < num_point; j++){
			int count = 0;
			for (int k = 0; k < num_tree_per_point; k++){
				int index = i * num_point * num_tree_per_point + j * num_tree_per_point + k;
				cv::Mat_<float> tmp = m_Model.__rf.row(index);
				
				int pos = count;
				for (int l = 0; l < tmp.cols; l += node_step, pos++){
					m_AX[i](j, pos) = tmp(0, l);
					m_AY[i](j, pos) = tmp(0, l + 1);
					m_BX[i](j, pos) = tmp(0, l + 2);
					m_BY[i](j, pos) = tmp(0, l + 3);
					m_Thresh[i](j, pos) = tmp(0, l + 4);
				}
				count += num_node;
  			}
		}
	}

	cv::Mat_<int> tmp1 = cv::Mat::zeros(num_node, 1, CV_32SC1);
	cv::Mat_<int> tmp2 = cv::Mat::zeros(num_node, 2, CV_32SC1);

	for (int i = num_node / 2; i < num_node; i++){
		tmp1(i, 0) = 1;
	}

	int tmp3 = 1;
	for (int i = 0; i < num_node / 2; i++){
		tmp2(i, 0) = tmp3++;
		tmp2(i, 1) = tmp3++;
	}	

	for (int i = 0; i < num_tree_per_point; i++){
		m_Isleaf.push_back(tmp1);
		m_Cnodes.push_back(tmp2);
	} 	

	tmp3 = m_Model.__head.__num_leaf - 1;
	m_Idleafnodes = cv::Mat::zeros(m_Model.__head.__num_leaf, 1, CV_32SC1);
	for (int i = 0; i < m_Idleafnodes.rows; i++){
		m_Idleafnodes(i, 0) = tmp3++;
	}		
}

void cModel::UpDate(cv::Mat_<int>&binary, cv::Rect &bbox, cv::Mat_<float>& shape, int stage)
{	
	int num_point = m_Model.__head.__num_point;
	int w_cols = 2 * num_point;
	int w_rows = binary.cols;	

	cv::Mat_<float> deltashapes = cv::Mat::zeros(1, w_cols, CV_32FC1);

	for (int i = 0; i < w_rows; i++){
		if (binary(0, i) == 1){
			deltashapes += m_Model.__w.row(stage * w_rows + i);
		}
	}

	cv::Mat_<float> deltax = deltashapes.colRange(0, num_point).t();
	deltax *= bbox.width;
	cv::Mat_<float> deltay = deltashapes.colRange(num_point, w_cols).t();
	deltay *= bbox.height;	
		
	shape.col(0) += deltax;
	shape.col(1) += deltay;
}

cv::Mat_<float> cModel::Reshape(cv::Mat_<float>& mean, cv::Rect& faceBox)
{
	cv::Mat_<double> modelShape = mean.clone();
	cv::Mat_<double> xCoords = modelShape.colRange(0, modelShape.cols / 2);
	cv::Mat_<double> yCoords = modelShape.colRange(modelShape.cols / 2, modelShape.cols);

	double minX, maxX, minY, maxY;
	cv::minMaxLoc(xCoords, &minX, &maxX);//得到x的最大/最小值
	cv::minMaxLoc(yCoords, &minY, &maxY);//得到y的最大/最小值
	double faceboxScaleFactor = m_Params->__facebox_scale_factor;
	double modelWidth = maxX - minX;
	double modelHeight = maxY - minY;

	// scale it:
	modelShape = modelShape * (faceBox.width / modelWidth + faceBox.height / modelHeight) / (m_Params->__facebox_scale_const * faceboxScaleFactor);
	// translate the model:
	cv::Scalar meanX = cv::mean(xCoords);
	double meanXd = meanX[0];
	cv::Scalar meanY = cv::mean(yCoords);
	double meanYd = meanY[0];
	// move it:
	xCoords += faceBox.x + faceBox.width / m_Params->__facebox_width_div - meanXd;
	yCoords += faceBox.y + faceBox.height / m_Params->__facebox_height_div - meanYd;
	return modelShape;
}

cv::Mat_<float> cModel::Reshape_alt(cv::Mat_<float>& mean, cv::Rect& faceBox)
{
	cv::Mat_<double> modelShape = mean.clone();
	cv::Mat_<double> xCoords = modelShape.colRange(0, modelShape.cols / 2);
	cv::Mat_<double> yCoords = modelShape.colRange(modelShape.cols / 2, modelShape.cols);

	double minX, maxX, minY, maxY;
	cv::minMaxLoc(xCoords, &minX, &maxX);//得到x的最大/最小值
	cv::minMaxLoc(yCoords, &minY, &maxY);//得到y的最大/最小值
	double faceboxScaleFactor = m_Params->__facebox_scale_factor;
	double modelWidth = maxX - minX;
	double modelHeight = maxY - minY;

	xCoords -= minX;
	yCoords -= minY;

	// scale it:
	xCoords *= faceBox.width / modelWidth;
	yCoords *= faceBox.height / modelHeight;

	// modelShape = modelShape * (faceBox.width / modelWidth + faceBox.height / modelHeight) / (params->__facebox_scale_const * faceboxScaleFactor);
	// translate the model:
	// cv::Scalar meanX = cv::mean(xCoords);
	// double meanXd = meanX[0];
	// cv::Scalar meanY = cv::mean(yCoords);
	// double meanYd = meanY[0];
	// move it:
	xCoords += faceBox.x; // +faceBox.width / params->__facebox_width_div - meanXd;
	yCoords += faceBox.y; // +faceBox.height / params->__facebox_height_div - meanYd;
	return modelShape;
}
