#include "globalregression.h"
#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <stdlib.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {}

cRegression::cRegression(sTrainData* data, sParams* params)
{
	m_TrainData = data;
	m_Params = params;		
	m_Evaluate = new cEvaluate;
}


cv::Mat_<double> cRegression::GlobalRegression(const cv::Mat_<int>&binaryfeatures, int stage)
{
	cv::Mat_<double> deltashapes = cv::Mat::zeros(m_TrainData->__dbsizes, 2 * m_TrainData->__numlandmarks, CV_64FC1);
	std::cout << "compute residual." << std::endl;
	for (int i = 0;i < m_TrainData->__dbsizes; i++){
		cv::Mat_<double> residua_col0 = m_TrainData->__data[i].__shapes_residual.col(0).t();
		cv::Mat_<double> residua_col1 = m_TrainData->__data[i].__shapes_residual.col(1).t();

		residua_col0.copyTo(deltashapes.row(i).colRange(0, m_TrainData->__numlandmarks));
		residua_col1.copyTo(deltashapes.row(i).colRange(m_TrainData->__numlandmarks, 2 * m_TrainData->__numlandmarks));
	}		
	std::cout << "compute residual finished." << std::endl;

	std::cout << "linear regression." << std::endl;
	cv::Mat_<double> W_liblinear = cv::Mat::zeros(binaryfeatures.cols, deltashapes.cols,CV_64FC1);
	// std::cout << "convert to sparse feature." << std::endl;
	cv::SparseMat_<int> binaryfeatures_sparse(binaryfeatures);
	// std::cout << binaryfeatures_sparse.size(1) << std::endl;
	// std::cout << "convert to sparse feature finished." << std::endl;
    #pragma omp parallel for
	for (int i = 0; i < deltashapes.cols; i++){
		cv::Mat_<double> tmp = __train_regressor(deltashapes.col(i), binaryfeatures_sparse);
		tmp.copyTo(W_liblinear.col(i));
	}
	std::cout << "linear regression finished." << std::endl;

	cv::Mat_<double> binaryfeatures_d;
	binaryfeatures.convertTo(binaryfeatures_d,CV_64FC1);
	cv::Mat_<double> deltashapes_bar = binaryfeatures_d * W_liblinear;

	cv::Mat_<double> deltashapes_bar_x = deltashapes_bar.colRange(0, deltashapes_bar.cols/2);//54 X 68
	cv::Mat_<double> deltashapes_bar_y = deltashapes_bar.colRange(deltashapes_bar.cols / 2, deltashapes_bar.cols);

	std::vector<cv::Mat_<double>> preshapes(0);
	std::vector<cv::Mat_<double>> gtshapes(0);

	std::cout << "update stage." << std::endl;
	for (int i=0; i < m_TrainData->__dbsizes; i++){		
		cv::Mat_<double> delta_shape_interm_coord = cv::Mat::zeros(deltashapes_bar_x.cols,2,CV_64FC1);//68 X 2
		cv::Mat_<double> deltashapes_bar_x_t, deltashapes_bar_y_t;

		deltashapes_bar_x_t = deltashapes_bar_x.t();
		deltashapes_bar_y_t = deltashapes_bar_y.t();

		deltashapes_bar_x_t.col(i).copyTo(delta_shape_interm_coord.col(0));
		deltashapes_bar_y_t.col(i).copyTo(delta_shape_interm_coord.col(1));		

		delta_shape_interm_coord.col(0) *= m_TrainData->__data[i].__intermediate_bboxes[stage].width;
		delta_shape_interm_coord.col(1) *= m_TrainData->__data[i].__intermediate_bboxes[stage].height;	

		cv::Mat_<double> shape_newstage = m_TrainData->__data[i].__intermediate_shapes[stage] + delta_shape_interm_coord;		

		m_TrainData->__data[i].__intermediate_shapes[stage + 1] = shape_newstage.clone();
		m_TrainData->__data[i].__intermediate_bboxes[stage + 1] = m_TrainData->__data[i].__intermediate_bboxes[stage];

		preshapes.push_back(m_TrainData->__data[i].__intermediate_shapes[stage + 1].clone());
		gtshapes.push_back(m_TrainData->__data[i].__shape_gt.clone());

		/*
		cv::Mat img = m_TrainData->__data[i].__img_gray.clone();
		for (int k = 0; k < preshapes[i].rows; ++k) {
			cv::circle(img, cv::Point(m_TrainData->__data[i].__intermediate_shapes[stage](k, 0), 
				m_TrainData->__data[i].__intermediate_shapes[stage](k, 1)), 
				1, CV_RGB(128, 128, 128), 1);
			cv::circle(img, cv::Point(preshapes[i](k, 0), preshapes[i](k, 1)), 1, CV_RGB(255, 255, 255), 1);
			cv::circle(img, cv::Point(gtshapes[i](k, 0), gtshapes[i](k, 1)), 1, CV_RGB(0, 0, 0), 1);
		}
		cv::rectangle(img, m_TrainData->__data[i].__bbox_gt, CV_RGB(255, 255, 255), 1);
		cv::imshow("img", img);
		cv::waitKey(0);
		*/

		//calculate transform
		/*cv::Mat_<double> shape_newstage_x = shape_newstage.col(0);
		cv::Mat_<double> shape_newstage_y = shape_newstage.col(1);
		cv::Scalar cx = cv::mean(shape_newstage_x);
		cv::Scalar cy = cv::mean(shape_newstage_y);
		shape_newstage_x = shape_newstage_x - cx[0];
		shape_newstage_y = shape_newstage_y - cy[0];		

		cv::Rect         bbox   = m_TrainData->__data[i].__intermediate_bboxes[stage + 1];
		cv::Mat_<double> mean = m_Utils.Reshape(m_TrainData->__meanface_one_row, bbox, m_Params);
		cv::Mat_<double> mean_x = mean.colRange(0, mean.cols / 2).t();
		cv::Mat_<double> mean_y = mean.colRange(mean.cols / 2, mean.cols).t();

		cv::Scalar mx = cv::mean(mean_x);
		cv::Scalar my = cv::mean(mean_y);
		mean_x = mean_x - mx[0];
		mean_y = mean_y - my[0];
		cv::Mat_<double> mean_newstage = cv::Mat::zeros(mean_x.rows,2,CV_64FC1);
		mean_x.copyTo(mean_newstage.col(0));
		mean_y.copyTo(mean_newstage.col(1));

		m_TrainData->__data[i].__tf2meanshape = m_Utils.Calc_Point_Transform(shape_newstage, mean_newstage);
		m_TrainData->__data[i].__meanshape2tf = m_Utils.Calc_Point_Transform(mean_newstage, shape_newstage);*/		

		cv::Mat_<double> residual_col0, residual_col1, residual;
		residual = m_TrainData->__data[i].__shape_gt - shape_newstage;
		
		residual.col(0) = residual.col(0) / (double)m_TrainData->__data[i].__intermediate_bboxes[stage + 1].width;
		residual.col(1) = residual.col(1) / (double)m_TrainData->__data[i].__intermediate_bboxes[stage + 1].height;	

		//residual.col(0) = residual.col(0) * m_TrainData->__data[i].__tf2meanshape.at<double>(0, 0) + residual.col(1) * m_TrainData->__data[i].__tf2meanshape.at<double>(0, 1) + m_TrainData->__data[i].__tf2meanshape.at<double>(0, 2);
		//residual.col(1) = residual.col(1) * m_TrainData->__data[i].__tf2meanshape.at<double>(1, 0) + residual.col(1) * m_TrainData->__data[i].__tf2meanshape.at<double>(1, 1) + m_TrainData->__data[i].__tf2meanshape.at<double>(1, 2);

		residual.copyTo(m_TrainData->__data[i].__shapes_residual);		
	}
	std::wcout << "update stage finished." << std::endl;

	std::cout << "Mean Square Root Error: " << m_Evaluate->compute_error(preshapes, gtshapes) << std::endl;

	// std::cout << "regression stage finished." << std::endl;
	return W_liblinear;
}

cv::Mat_<double> cRegression::__train_regressor(const cv::Mat_<double>& label_vec, const cv::SparseMat_<int>& instance_mat)
{
	void(*print_func)(const char*) = &print_null;
	const char *error_msg;

	struct parameter param;
	struct problem   problem;
	struct feature_node *x_space = NULL;

	srand(1);
	// std::cout << "initialize liblinear parameter." << std::endl;
	param.solver_type = L2R_L2LOSS_SVR_DUAL;
	param.C = 1.0 / (double)label_vec.rows;
	param.eps = 0.1;
	param.p = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	// std::cout << "initialize liblinear parameter finished." << std::endl;
	set_print_string_function(print_func);

	std::vector<int>*  prob_x = NULL;
	prob_x = new std::vector<int>[label_vec.rows]; // number of samples = label_vec.rows

	cv::SparseMatConstIterator_<int> it, end;
	// std::cout << "copy feature." << std::endl;
	for (it = instance_mat.begin(), end = instance_mat.end(); it != end; ++it){
		const cv::SparseMat::Node* node = it.node();
		prob_x[node->idx[0]].push_back(node->idx[1]);
	}
	// std::cout << "copy feature finished." << std::endl;

	//sort the vector
	for (int i = 0; i < label_vec.rows; i++){
		std::sort(prob_x[i].begin(), prob_x[i].end());
	}
	
	problem.l = label_vec.rows;
	problem.n = instance_mat.size(1);
	problem.bias = -1;

	size_t nzcount = instance_mat.nzcount();
	int elements = (int)(nzcount + problem.l);	
	
	problem.y = Malloc(double, problem.l);
	problem.x = Malloc(struct feature_node *, problem.l);
	x_space   = Malloc(struct feature_node, elements);

	int j = 0;
	for (int i = 0; i < problem.l; i++){
		problem.y[i] = label_vec(i, 0);
		problem.x[i] = &x_space[j];
		
		for (int k = 0; k < prob_x[i].size(); k++){
			x_space[j].index = prob_x[i][k] + 1;
			x_space[j].value = 1;
			j++;
		}
		x_space[j++].index = -1;
	}

	delete[] prob_x;

	error_msg = check_parameter(&problem, &param);
	if (error_msg){
		fprintf(stderr, "ERROR: %s\n", error_msg);		
	}

	// std::cout << "train model." << std::endl;
	struct model *model = NULL;
	model = train(&problem, &param);
	// std::cout << "train model finished." << std::endl;

	cv::Mat_<double> weight = cv::Mat::zeros(model->nr_feature, 1, CV_64FC1);
	for (int i = 0; i < model->nr_feature; i++){
		weight(i, 0) = model->w[i];
	}

	free_and_destroy_model(&model);
	destroy_param(&param);

	free((void*)(problem.y));
	free((void*)(problem.x));
	free((void*)(x_space));
	return weight;
}


cv::Mat_<double> cRegression::__train_regressor(const cv::Mat_<double>& label_vec, const cv::Mat_<int>& instance_mat)
{
	void(*print_func)(const char*) = &print_null;
	const char *error_msg;

	struct parameter param;
	struct problem   problem;
	struct feature_node *x_space = NULL;

	srand(1);
	// std::cout << "initialize liblinear parameter." << std::endl;
	param.solver_type = L2R_L2LOSS_SVR_DUAL;
	param.C = 1.0 / (double)label_vec.rows;
	param.eps = 0.1;
	param.p = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	// std::cout << "initialize liblinear parameter finished." << std::endl;
	set_print_string_function(print_func);

	std::vector<int>*  prob_x = NULL;
	prob_x = new std::vector<int>[label_vec.rows]; // number of samples = label_vec.rows

	size_t nzcount = 0;
	// std::cout << "copy feature." << std::endl;
	for (int i = 0; i < instance_mat.rows; ++i) {
	    for (int j = 0; j < instance_mat.cols; ++j) {
		    int elem = instance_mat(i, j);
			if (elem != 0) {
				prob_x[i].push_back(j);
				++nzcount;
			}
		}
	}
	// std::cout << "copy feature finished." << std::endl;

	//sort the vector
	for (int i = 0; i < label_vec.rows; i++){
		std::sort(prob_x[i].begin(), prob_x[i].end());
	}

	problem.l = label_vec.rows;
	problem.n = instance_mat.cols;
	problem.bias = -1;

	int elements = (int)(nzcount + problem.l);

	problem.y = Malloc(double, problem.l);
	problem.x = Malloc(struct feature_node *, problem.l);
	x_space = Malloc(struct feature_node, elements);

	int j = 0;
	for (int i = 0; i < problem.l; i++){
		problem.y[i] = label_vec(i, 0);
		problem.x[i] = &x_space[j];

		for (int k = 0; k < prob_x[i].size(); k++){
			x_space[j].index = prob_x[i][k] + 1;
			x_space[j].value = 1;
			j++;
		}
		x_space[j++].index = -1;
	}

	delete[] prob_x;

	error_msg = check_parameter(&problem, &param);
	if (error_msg){
		fprintf(stderr, "ERROR: %s\n", error_msg);
	}

	// std::cout << "train model." << std::endl;
	struct model *model = NULL;
	model = train(&problem, &param);
	// std::cout << "train model finished." << std::endl;

	cv::Mat_<double> weight = cv::Mat::zeros(model->nr_feature, 1, CV_64FC1);
	for (int i = 0; i < model->nr_feature; i++){
		weight(i, 0) = model->w[i];
		// std::cout << weight(i, 0) << " "; // std::endl;
	}


	free_and_destroy_model(&model);
	destroy_param(&param);

	free((void*)(problem.y));
	free((void*)(problem.x));
	free((void*)(x_space));
	return weight;
}
