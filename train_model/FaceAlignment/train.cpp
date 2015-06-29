#include "train.h"
#include <stdio.h>

int g_max_numfeats[] = { 1000, 1000, 1000, 500, 500, 500, 400, 400 };

cTrain::cTrain(sParams *params)
{
	m_IBug = new cIBug(params);	
	m_Params = params;
	m_BinayFeat = NULL;
	m_Regression = NULL;
	m_RFS = NULL;
}

cTrain::~cTrain()
{	
	if (m_RFS){
		for (int i = 0; i < m_Params->__max_numstage; i++){
			m_RF->ReleaseRF(&m_RFS[i]);
		}
		m_RFS = NULL;
	}

	if (m_Regression){
		delete m_Regression;
		m_Regression = NULL;
	}

	if (m_BinayFeat){
		delete m_BinayFeat;
		m_BinayFeat = NULL;
	}

	if (m_RF){
		delete m_RF;
		m_RF = NULL;
	}

	if (m_IBug){
		delete m_IBug;
		m_IBug = NULL;
	}
}

void cTrain::Train()
{	
	
	__init_train_data();
	
	m_RF = new cRandomForest(&m_TrainData, m_Params);
	m_BinayFeat = new cBinaryFeat(&m_TrainData, m_Params);
	m_Regression = new cRegression(&m_TrainData, m_Params);
	m_RFS = new sRandomForest[m_Params->__max_numstage];
	cv::Mat_<double> weight, weights;	
	int64 time_beg, time_end;

	for (int i = 0; i < m_Params->__max_numstage; i++){
		std::string path_lbf = "binaryfeature_s_" + std::to_string(i) + ".xml";
		cv::FileStorage file_in(path_lbf, cv::FileStorage::READ);
		cv::Mat binaryfeatures;
		if (!file_in.isOpened()) {
			std::cout << "train random forest for train stage " << i << "." << std::endl;
			time_beg = GetTickCount64();
			m_RF->Train_Random_Forest(&m_RFS[i], i);
			time_end = GetTickCount64();
			std::cout << "time cost for train random forest: " << (time_end - time_beg) / 1000 << " secs." << std::endl;

			std::cout << "derive binary feature for train stage " << i << "." << std::endl;
			time_beg = GetTickCount64();
			binaryfeatures = m_BinayFeat->DerivBinaryfeat(&m_RFS[i], i);
			time_end = GetTickCount64();
			std::cout << "time cost for derive binary feature: " << (time_end - time_beg) / 1000 << " secs." << std::endl;

			cv::FileStorage file_out(path_lbf, cv::FileStorage::WRITE);
			file_out << "binaryfeature" << binaryfeatures;
			file_out.release();
		}
		else {
			file_in["binaryfeature"] >> binaryfeatures;
		}
		file_in.release();

		std::cout << "global regression for train stage " << i << "." << std::endl;
		time_beg = GetTickCount64();
		weight = m_Regression->GlobalRegression(binaryfeatures, i);
		time_end = GetTickCount64();
		std::cout << "time cost for global regression: " << (time_end - time_beg) / 1000 << " secs." << std::endl;

		weights.push_back(weight);
	}	

	std::cout << "rf convert to model." << std::endl;
	cv::Mat_<double> RF;
	__rf2model(m_RFS, RF);
	std::cout << "rf convert to model finished." << std::endl;

	cv::Mat_<float> RF_C, W_C;
	RF.convertTo(RF_C,CV_32FC1);
	weights.convertTo(W_C, CV_32FC1);
	__save_model(m_Params->__outputmodel, RF_C, W_C);
}

void cTrain::__save_model(std::string filename, cv::Mat_<float>&rf, cv::Mat_<float>&w)
{
	cv::Mat_<float> meanface;
	FILE *fp = NULL;
	m_TrainData.__meanface.convertTo(meanface, CV_32FC1);
	meanface = meanface.t();
	
	fp = fopen(filename.c_str(), "wb");
	if (fp == NULL){
		return;
	}

	fwrite(&m_Head, sizeof(m_Head), 1, fp);

	for (int i = 0; i < meanface.rows; i++){
		for (int j = 0; j < meanface.cols; j++){			
			fwrite(&meanface(i, j), sizeof(float), 1, fp);
		}
	}
	
	for (int i = 0; i < rf.rows; i++){		
		fwrite(rf.ptr<float>(i), sizeof(float)*rf.cols, 1, fp);
	}

	
	for (int i = 0; i < w.rows; i++){		
		fwrite(w.ptr<float>(i), sizeof(float)*w.cols, 1, fp);
	}

	fclose(fp);	
  }

void cTrain::__rf2model(const sRandomForest *rf, cv::Mat_<double>& RF)
{	
	int num_stage = m_Params->__max_numstage;
	int num_point = m_TrainData.__numlandmarks;
	int num_tree_per_point = m_Params->__max_numtrees;
	int tree_depth = m_Params->__max_depth - 1;
	int node_step = 5;

	int num_node = (int)(pow(2.0f, tree_depth) - 1);
	int num_leaf = num_node + 1;
	int dim_tree = node_step*num_node;

	int num_tree_per_stage = num_point*num_tree_per_point;
	int num_tree_total     = num_stage*num_point*num_tree_per_point;
	int dim_feat           = num_leaf*num_tree_per_stage;

	m_Head.__num_stage = num_stage;
	m_Head.__num_point = num_point;
	m_Head.__num_tree_per_point = num_tree_per_point;
	m_Head.__tree_depth = tree_depth;
	m_Head.__node_step = node_step;

	m_Head.__num_node = num_node;
	m_Head.__num_leaf = num_leaf;
	m_Head.__dim_tree = dim_tree;

	m_Head.__num_tree_per_stage = num_tree_per_stage;
	m_Head.__num_tree_total = num_tree_total;
	m_Head.__dim_feat = dim_feat;

	
	RF = cv::Mat::zeros(num_tree_total, dim_tree , CV_64FC1);
	std::cout << "assign model" << std::endl;
	for (int stage = 0; stage < num_stage; stage++){
		
		for (int p = 0; p < num_point; p++){
			for (int t = 0; t < num_tree_per_point; t++){
				sRF* Tree = &(rf[stage].__RF[p][t]);
				cv::Mat_<double> anglepairs = Tree->__feat.colRange(0, 2);
				cv::Mat_<double> radiuspairs = Tree->__feat.colRange(2, 4);

				cv::Mat_<double> angles_cos, angles_sin;

				angles_cos = cv::Mat::zeros(anglepairs.rows, 2, CV_64FC1);
				angles_sin = cv::Mat::zeros(anglepairs.rows, 2, CV_64FC1);

				for (int i = 0; i < radiuspairs.rows; i++){
					angles_cos(i, 0) = cos(anglepairs(i, 0));
					angles_cos(i, 1) = cos(anglepairs(i, 1));
					angles_sin(i, 0) = sin(anglepairs(i, 0));
					angles_sin(i, 1) = sin(anglepairs(i, 1));
				}

				cv::Mat_<double> ax, ay, bx, by;

				cv::multiply(angles_cos.col(0), radiuspairs.col(0), ax);
				cv::multiply(angles_sin.col(0), radiuspairs.col(0), ay);
				ax *= m_Params->__max_raio_radius[stage];
				ay *= m_Params->__max_raio_radius[stage];

				cv::multiply(angles_cos.col(1), radiuspairs.col(1), bx);
				cv::multiply(angles_sin.col(1), radiuspairs.col(1), by);
				bx *= m_Params->__max_raio_radius[stage];
				by *= m_Params->__max_raio_radius[stage];

				cv::Mat_<double> ax_x, ay_y, bx_x, by_y, th;
				ax_x = ax.rowRange(0, Tree->__num_leafnodes - 1);
				ay_y = ay.rowRange(0, Tree->__num_leafnodes - 1);
				bx_x = bx.rowRange(0, Tree->__num_leafnodes - 1);
				by_y = by.rowRange(0, Tree->__num_leafnodes - 1);
				th = Tree->__thresh.rowRange(0, Tree->__num_leafnodes - 1);

				cv::Mat_<double> temp = cv::Mat::zeros(Tree->__num_leafnodes - 1, 5, CV_64FC1);
				
				ax_x.copyTo(temp.col(0));
				ay_y.copyTo(temp.col(1));
				bx_x.copyTo(temp.col(2));
				by_y.copyTo(temp.col(3));
				th.copyTo(temp.col(4));			

				temp = temp.reshape(0, 1);

				int  k = stage * num_tree_per_stage + p * num_tree_per_point + t;
				temp.copyTo(RF.row(k));
			}
		}		
	}	
}

void cTrain::__init_train_data()
{
	std::cout << "get data..." << std::endl;
	m_IBug->GetMaterial();	
	m_TrainData = m_IBug->GetTrainData();
	std::cout << "dbsize:" << m_TrainData.__dbsizes<< std::endl;

	int num_landmarks = m_TrainData.__numlandmarks;

	cv::Mat_<double> meanface = __calcmeanface();
	m_TrainData.__meanface_one_row = meanface.clone();

	cv::Mat_<double> face_col0 = meanface.colRange(0, num_landmarks).t();
	cv::Mat_<double> face_col1 = meanface.colRange(num_landmarks, num_landmarks * 2).t();
	m_TrainData.__meanface = cv::Mat::zeros(num_landmarks, 2, CV_64FC1);
	face_col0.copyTo(m_TrainData.__meanface.col(0));
	face_col1.copyTo(m_TrainData.__meanface.col(1));

	for (int i = 0; i < m_TrainData.__dbsizes; i++){
		
		cv::Mat_<double> intermediate_shapes = cv::Mat::zeros(num_landmarks, 2, CV_64FC1);
		cv::Mat_<double> tmp = m_Utils.Reshape_alt(meanface, m_TrainData.__data[i].__bbox_gt,m_Params);
		cv::Mat_<double> tmp_col0 = tmp.colRange(0, num_landmarks).t();
		cv::Mat_<double> tmp_col1 = tmp.colRange(num_landmarks, num_landmarks * 2).t();
		tmp_col0.copyTo(intermediate_shapes.col(0));
		tmp_col1.copyTo(intermediate_shapes.col(1));		

		m_TrainData.__data[i].__intermediate_shapes[0] = intermediate_shapes.clone();
		m_TrainData.__data[i].__intermediate_bboxes[0] = m_TrainData.__data[i].__bbox_gt;

		
		cv::Mat_<uchar> img = m_TrainData.__data[i].__img_gray.clone();
		for (int k = 0; k < num_landmarks; ++k) {
			cv::circle(img, cv::Point(intermediate_shapes(k, 0), intermediate_shapes(k, 1)), 1, CV_RGB(255, 255, 255), 1);
		}
		cv::rectangle(img, m_TrainData.__data[i].__bbox_gt, CV_RGB(255, 255, 255), 1);
		cv::imshow("img",img);
		cv::waitKey(0);
		

		//calculate transform
		/*cv::Mat_<double> intermediate_x = intermediate_shapes.col(0);
		cv::Mat_<double> intermediate_y = intermediate_shapes.col(1);
		cv::Scalar cx = cv::mean(intermediate_x);
		cv::Scalar cy = cv::mean(intermediate_y);
		intermediate_x = intermediate_x - cx[0];
		intermediate_y = intermediate_y - cy[0];
		m_TrainData.__data[i].__tf2meanshape = m_Utils.Calc_Point_Transform(intermediate_shapes, intermediate_shapes);
		m_TrainData.__data[i].__meanshape2tf = m_Utils.Calc_Point_Transform(intermediate_shapes, intermediate_shapes);*/
		
		cv::Mat_<double> minus = m_TrainData.__data[i].__shape_gt - intermediate_shapes;

		m_TrainData.__data[i].__shapes_residual = cv::Mat::zeros(intermediate_shapes.rows,2,CV_64FC1);		
		
		minus.col(0) = minus.col(0) / (double)m_TrainData.__data[i].__bbox_gt.width;
		minus.col(1) = minus.col(1) / (double)m_TrainData.__data[i].__bbox_gt.height;		
		minus.copyTo(m_TrainData.__data[i].__shapes_residual);			
	}
}

cv::Mat_<double> cTrain::__calcmeanface()
{
	cv::Mat_<double> meanface;
	int num_landmarks = m_TrainData.__numlandmarks;
	cv::Mat_<double> landmarks = cv::Mat::zeros(m_TrainData.__dbsizes, num_landmarks * 2, CV_64FC1);

	std::cout << "calc meanface..." << std::endl;

	std::vector<cv::Rect> facebox;

	for (int i = 0; i < m_TrainData.__dbsizes; i++){
		cv::Mat_<double> tmp1 = m_TrainData.__data[i].__shape_gt;

		cv::Mat_<double> tmp2 = tmp1.col(0).t();
		tmp2.copyTo(landmarks.row(i).colRange(0, num_landmarks));

		cv::Mat_<double> tmp3 = tmp1.col(1).t();
		tmp3.copyTo(landmarks.row(i).colRange(num_landmarks, num_landmarks * 2));

		facebox.push_back(m_TrainData.__data[i].__bbox_gt);
	}

	meanface = __procrustes(landmarks, facebox);

	cv::Mat_<double> facerescale = cv::Mat::zeros(landmarks.rows, landmarks.cols, CV_64FC1);	
	for (int i = 0; i < landmarks.rows; i++){		
		cv::Mat_<double> tmp = __align(meanface, m_TrainData.__data[i].__bbox_gt);
		tmp.copyTo(facerescale.row(i));		
		
	}

	Statistics satistics = __statistics(facebox, landmarks, facerescale);
	__rescale(meanface, satistics);

	std::cout << "done!" << std::endl;
	return meanface;
}

cv::Mat_<double> cTrain::__procrustes(const cv::Mat_<double>& groundtruth, const std::vector<cv::Rect>& faceboxes)
{
	int num_landmarks = groundtruth.cols / 2;
	cv::Mat_<double> P = groundtruth.clone();
	int num_images = groundtruth.rows;
	//每个点分别除以width和height
	for (int i = 0; i < num_images; i++){
		cv::Mat_<double> gt = P.row(i);
		cv::Mat_<double> gt_x = gt.colRange(0, num_landmarks);
		cv::Mat_<double> gt_y = gt.colRange(num_landmarks, 2 * num_landmarks);
		gt_x = gt_x / (double)faceboxes[i].width;
		gt_y = gt_y / (double)faceboxes[i].height;
		//将形状移动到重心
		cv::Scalar cx = cv::mean(gt_x);
		cv::Scalar cy = cv::mean(gt_y);

		gt_x = gt_x - cx[0];
		gt_y = gt_y - cy[0];
	}

	cv::Mat_<double> C_old;
	cv::Mat_<double> C, mean;
	for (int iter = 0; iter < m_Params->__procrustes_iters; iter++){
		cv::reduce(P, C, 0, CV_REDUCE_AVG); //求得平均值
		mean = C.clone();
		cv::normalize(C, C);//归一化

		if (iter > 0){
			if (norm(C, C_old) < m_Params->__procrustes_errors){
				break;
			}
		}

		C_old = C.clone();
		for (int i = 0; i < num_images; i++){
			cv::Mat_<double> R = __rot_scale_align(P.row(i), C);
			for (int j = 0; j < num_landmarks; j++){
				double x = P(i, j), y = P(i, j + num_landmarks);
				P(i, j) = R(0, 0) * x + R(0, 1) * y;
				P(i, j + num_landmarks) = R(1, 0) * x + R(1, 1) * y;
			}
		}
	}
	return mean;
}

cv::Mat_<double> cTrain::__rot_scale_align(const cv::Mat_<double> &src, const cv::Mat_<double> &dst)
{
	int n = src.cols / 2; 
	double a = 0.0, b = 0.0, d = 0.0;
	for (int i = 0; i < n; i++){
		d += src(0,i) * src(0,i) +src(0,i + n) * src(0,i + n);
		a += src(0,i) * dst(0,i) +src(0,i + n) * dst(0,i + n);
		b += src(0,i) * dst(0,i + n) - src(0,i + n) * dst(0,i);
	}
	a /= d; b /= d;
	return (cv::Mat_<double>(2, 2) << a, -b, b, a);
}

Statistics cTrain::__statistics(std::vector<cv::Rect>& faceboxes, const cv::Mat_<double>& landmarks, const cv::Mat_<double>& meanface)
{
	int num_images = landmarks.rows;
	cv::Mat_<double> _tx = cv::Mat::zeros(num_images, 1, CV_64FC1);
	cv::Mat_<double> _ty = cv::Mat::zeros(num_images, 1, CV_64FC1);
	cv::Mat_<double> _sx = cv::Mat::zeros(num_images, 1, CV_64FC1);
	cv::Mat_<double> _sy = cv::Mat::zeros(num_images, 1, CV_64FC1);
	int num_landmarks = landmarks.cols / 2;

	for (int i = 0; i < num_images; ++i) {
		cv::Rect bbox = faceboxes[i];
		
		cv::Mat groundtruth_x = landmarks.row(i).colRange(0, num_landmarks);
		cv::Mat groundtruth_y = landmarks.row(i).colRange(num_landmarks, num_landmarks * 2);

		cv::Mat meanface_x = meanface.row(i).colRange(0, num_landmarks);
		cv::Mat meanface_y = meanface.row(i).colRange(num_landmarks, num_landmarks * 2);
		
		_tx(i, 0) = __calctranslation(groundtruth_x, meanface_x) / bbox.width;
		_ty(i, 0) = __calctranslation(groundtruth_y, meanface_y) / bbox.height;
		
		_sx(i, 0) = __calcscaleratio(groundtruth_x, meanface_x);
		_sy(i, 0) = __calcscaleratio(groundtruth_y, meanface_y);
	}
	
	Statistics statistics;	

	statistics._tx = cv::mean(_tx).val[0];
	statistics._ty = cv::mean(_ty).val[0];
	statistics._sx = cv::mean(_sx).val[0];
	statistics._sy = cv::mean(_sy).val[0];
	return statistics;
}

double cTrain::__calctranslation(const cv::Mat_<double>& landmark, const cv::Mat_<double>& initshape)
{
	cv::Scalar mean1 = cv::mean(landmark);
	cv::Scalar mean2 = cv::mean(initshape);	
	return (mean2[0] - mean1[0]);
}

double cTrain::__calcscaleratio(const cv::Mat_<double>& landmark, const cv::Mat_<double>& initshape)
{	
	double land_min, land_max;
	cv::minMaxIdx(landmark, &land_min, &land_max);
	double init_min, init_max;
	cv::minMaxIdx(initshape, &init_min, &init_max);
	return (init_max - init_min) / (land_max - land_min);
}

cv::Mat_<double> cTrain::__align(const cv::Mat_<double>& mean, const cv::Rect& faceBox)
{
	cv::Mat_<double> mean_shape = mean.clone();
	cv::Mat_<double> mean_shape_x = mean_shape.colRange(0, mean_shape.cols / 2);
	cv::Mat_<double> mean_shape_y = mean_shape.colRange(mean_shape.cols / 2, mean_shape.cols);

	mean_shape_x = (mean_shape_x + 0.5) * faceBox.width + faceBox.x;
	mean_shape_y = (mean_shape_y + 0.5) * faceBox.height + faceBox.y;

	return mean_shape;
}

void  cTrain::__rescale(cv::Mat_<double>& mean, const Statistics & statistics)
{
	cv::Mat mean_x = mean.colRange(0, mean.cols / 2);
	cv::Mat mean_y = mean.colRange(mean.cols / 2, mean.cols);
	mean_x = (mean_x - statistics._tx) / statistics._sx;
	mean_y = (mean_y - statistics._ty) / statistics._sy;
}