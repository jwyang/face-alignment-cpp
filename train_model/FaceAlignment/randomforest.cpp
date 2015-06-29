#include "randomforest.h"
#include <math.h>
#include <iostream>
#include <float.h>

#include <boost/random/random_device.hpp>
#include <boost/random.hpp>
#include <boost/range/algorithm/random_shuffle.hpp>
#include <boost/lexical_cast.hpp>

#include <fstream>

//radius and angles
double g_radius[] ={   0.0, 0.0333333333333333, 0.0666666666666667, 0.1,0.133333333333333,\
	0.166666666666667, 0.2, 0.233333333333333,  0.266666666666667,  0.300000000000000, 0.333333333333333, \
	0.366666666666667, 0.4, 0.433333333333333,  0.466666666666667,  0.500000000000000, 0.533333333333333, \
	0.566666666666667, 0.6, 0.633333333333333,  0.666666666666667,  0.700000000000000, 0.733333333333333, \
	0.766666666666667, 0.8, 0.833333333333333,  0.866666666666667,  0.900000000000000, 0.933333333333333, \
	0.966666666666667, 1.0 \
};
double g_angles[] ={ \
	0.0, 0.174532925199433, 0.349065850398866, 0.523598775598299, 0.698131700797732, 0.872664625997165, \
	1.04719755119660, 1.22173047639603, 1.39626340159546, 1.57079632679490, 1.74532925199433, 1.91986217719376, \
	2.09439510239320, 2.26892802759263, 2.44346095279206, 2.61799387799149, 2.79252680319093, 2.96705972839036, \
	3.14159265358979, 3.31612557878923, 3.49065850398866, 3.66519142918809, 3.83972435438753, 4.01425727958696, \
	4.18879020478639, 4.36332312998582, 4.53785605518526, 4.71238898038469, 4.88692190558412, 5.06145483078356, \
	5.23598775598299, 5.41052068118242, 5.58505360638185, 5.75958653158129, 5.93411945678072, 6.10865238198015, \
	6.28318530717959 \
};

cRandomForest::cRandomForest(sTrainData  *data, sParams *params)
{	
	m_TrainData = data;
	m_Params = params;		
	__init();
}

cRandomForest::~cRandomForest()
{
	if (m_Data){
		delete[] m_Data;
	}	
}

void cRandomForest::ReleaseRF(sRandomForest* rf)
{
	if (rf && rf->__RF){
		for (int i = 0; i < rf->__rows; i++){
			for (int j = 0; j < rf->__cols; j++){
				if (rf->__RF[i][j].__ind_samples){
					delete[] rf->__RF[i][j].__ind_samples;
				}
			}

			if (rf->__RF[i]){
				delete[] rf->__RF[i];
			}
		}
		delete[] rf->__RF;
	}
}

void cRandomForest::__init()
{
	m_Q = floor((double)m_TrainData->__dbsizes / ((1.0 - m_Params->__bagging_overlap) * m_Params->__max_numtrees));
	m_Data = new cv::Mat_<int>[m_Params->__max_numtrees];

	for (int i = 0; i < m_Params->__max_numtrees; i++){
		int is = (int)MAX(floor(i * m_Q - i * m_Q * m_Params->__bagging_overlap), 0);
		int ie = (int)MIN(is + m_Q, m_TrainData->__dbsizes - 1);
		int length = ie - is + 1;
		 
		m_Data[i] = cv::Mat::zeros(1, length, CV_32SC1);

		int k = 0;
		for (int j = is; k < length; k++, j++){
			m_Data[i](0, k) = j;
		}		
	}

}

void  cRandomForest::Train_Random_Forest(sRandomForest* random_forest, int stage)
{
	try{
		random_forest->__RF = new sRF*[m_TrainData->__numlandmarks]; //68个点

		for (int i = 0; i < m_TrainData->__numlandmarks; i++){ //每个点10棵树
			random_forest->__RF[i] = new sRF[m_Params->__max_numtrees];
		}
	}catch (std::exception e){
		std::cout << e.what() << std::endl;
	}

	random_forest->__rows = m_TrainData->__numlandmarks;
	random_forest->__cols = m_Params->__max_numtrees;
	// omp_set_num_threads(4);
    #pragma omp parallel for
	for (int i = 0; i < m_TrainData->__numlandmarks; i++){
		// std::cout << "landmark: [" << i << "]" << std::endl;
		int64 time_beg = GetTickCount64();
		for (int t = 0; t < m_Params->__max_numtrees; t++){
			// std::cout << "tree: [" << t << "]" << std::endl;
			int is = (int)MAX(floor(t * m_Q - t * m_Q * m_Params->__bagging_overlap), 0);
			int ie = (int)MIN(is + m_Q, m_TrainData->__dbsizes - 1);
			int length = ie - is + 1; 			
			
			random_forest->__RF[i][t].__ind_samples = new cv::Mat_<int>[m_Params->__max_nodes];
			random_forest->__RF[i][t].__issplit     = cv::Mat::zeros(m_Params->__max_nodes, 1, CV_32SC1);
			random_forest->__RF[i][t].__pnode       = cv::Mat::zeros(m_Params->__max_nodes, 1, CV_32SC1);
			random_forest->__RF[i][t].__depth       = cv::Mat::zeros(m_Params->__max_nodes, 1, CV_32SC1);
			random_forest->__RF[i][t].__cnodes      = cv::Mat::zeros(m_Params->__max_nodes, 2, CV_32SC1);
			random_forest->__RF[i][t].__isleafnode  = cv::Mat::zeros(m_Params->__max_nodes, 1, CV_32SC1);
			random_forest->__RF[i][t].__feat        = cv::Mat::zeros(m_Params->__max_nodes, 4, CV_64FC1);
			random_forest->__RF[i][t].__thresh      = cv::Mat::zeros(m_Params->__max_nodes, 1, CV_64FC1);
			
			random_forest->__RF[i][t].__ind_samples[0] = cv::Mat::zeros(1, length, CV_32SC1);

			for (int j = 0; j < length; j++){
				random_forest->__RF[i][t].__ind_samples[0](0, j) = j;
			}

			random_forest->__RF[i][t].__depth(0, 0)      = 1;
			random_forest->__RF[i][t].__isleafnode(0, 0) = 1;

			int num_nodes     = 1;
			int num_leafnodes = 1;
			bool stop = false;

			while (!stop){
				int num_nodes_iter = num_nodes;
				int num_split = 0;

				for (int n = 0; n < num_nodes_iter; n++){
					if (!random_forest->__RF[i][t].__issplit(n, 0)){
						if (random_forest->__RF[i][t].__depth(n, 0) == m_Params->__max_depth){							
							random_forest->__RF[i][t].__issplit(n, 0) = 1;
						}else{
							sRT rt;
							__splitnode(i, stage, t, random_forest->__RF[i][t].__ind_samples[n], &rt);
							
							rt.__feat.copyTo(random_forest->__RF[i][t].__feat.row(n));						
							random_forest->__RF[i][t].__thresh(n, 0) = rt.__thresh;
							random_forest->__RF[i][t].__issplit(n, 0) = 1;
							random_forest->__RF[i][t].__cnodes(n, 0) = num_nodes;
							random_forest->__RF[i][t].__cnodes(n, 1) = num_nodes + 1;
							random_forest->__RF[i][t].__isleafnode(n, 0) = 0;

							if (!rt.__lcind.empty()){
								random_forest->__RF[i][t].__ind_samples[num_nodes] = rt.__lcind.clone();
							}
							
							random_forest->__RF[i][t].__issplit(num_nodes, 0) = 0;
							random_forest->__RF[i][t].__pnode(num_nodes, 0) = n;
							random_forest->__RF[i][t].__depth(num_nodes, 0) = random_forest->__RF[i][t].__depth(n, 0) + 1;
							random_forest->__RF[i][t].__cnodes(num_nodes, 0) = 0;
							random_forest->__RF[i][t].__cnodes(num_nodes, 1) = 0;
							random_forest->__RF[i][t].__isleafnode(num_nodes, 0) = 1;

							if (!rt.__rcind.empty()){
								random_forest->__RF[i][t].__ind_samples[num_nodes + 1] = rt.__rcind.clone();
							}
							
							random_forest->__RF[i][t].__issplit(num_nodes + 1, 0) = 0;
							random_forest->__RF[i][t].__pnode(num_nodes + 1, 0) = n;
							random_forest->__RF[i][t].__depth(num_nodes + 1, 0) = random_forest->__RF[i][t].__depth(n, 0) + 1;
							random_forest->__RF[i][t].__cnodes(num_nodes + 1, 0) = 0;
							random_forest->__RF[i][t].__cnodes(num_nodes + 1, 1) = 0;
							random_forest->__RF[i][t].__isleafnode(num_nodes + 1, 0) = 1;

							num_split = num_split + 1;
							num_leafnodes = num_leafnodes + 1;
							num_nodes = num_nodes + 2;
						}
					}
				}//end num_nodes_iter

				if (num_split == 0){
					stop = true;
				}else{
					random_forest->__RF[i][t].__num_leafnodes = num_leafnodes;
					random_forest->__RF[i][t].__num_nodes = num_nodes;
				}
			}// end while

			for (int rows = 0; rows < random_forest->__RF[i][t].__isleafnode.rows; rows++){
				if (random_forest->__RF[i][t].__isleafnode(rows, 0) == 1){
					cv::Mat_<int> tmp = cv::Mat::zeros(1, 1, CV_32SC1);
					tmp(0, 0) = rows;
					random_forest->__RF[i][t].__id_leafnodes.push_back(tmp);
				}
			}
		}//end max_numtrees
		int64 time_end = GetTickCount64();
		// std::cout << "time cost for " << i << "-th landmark: " << (time_end - time_beg) / 1000 << " secs." << std::endl;
	}//end land_mark

	std::cout << "training stage " << stage << " finished." << std::endl;
}

void cRandomForest::__splitnode(int lmarkID, int stage, int data_t,const cv::Mat_<int> &ind_samples, sRT* rt)
{
	if (ind_samples.empty()){
		rt->__thresh = 0.0;
		rt->__feat = cv::Mat::zeros(1, 4, CV_64FC1);
		return;
	}	

	cv::Mat_<double> radiuspairs, anglepairs, angles_cos, angles_sin;
	__getproposals(m_Params->__max_numfeats[stage], radiuspairs, anglepairs);	

	angles_cos = cv::Mat::zeros(anglepairs.rows, 2, CV_64FC1);
	angles_sin = cv::Mat::zeros(anglepairs.rows, 2, CV_64FC1);

	for (int i = 0; i < radiuspairs.rows; i++){
		angles_cos(i, 0) = cos(anglepairs(i, 0));
		angles_cos(i, 1) = cos(anglepairs(i, 1));
		angles_sin(i, 0) = sin(anglepairs(i, 0));
		angles_sin(i, 1) = sin(anglepairs(i, 1));
	}

	int length = ind_samples.cols;
	cv::Mat_<double> pdfeats = cv::Mat::zeros(m_Params->__max_numfeats[stage], length, CV_64FC1);
	cv::Mat_<double> shapes_residual = cv::Mat::zeros(length, 2, CV_64FC1);		

	for (int j = 0; j < length; j++){	
		
		int index = m_Data[data_t](0, ind_samples(0, j));
		cv::Mat_<double> pixel_a_x_imgcoord, pixel_a_y_imgcoord, pixel_b_x_imgcoord, pixel_b_y_imgcoord;

		cv::multiply(angles_cos.col(0), radiuspairs.col(0), pixel_a_x_imgcoord);
		cv::multiply(angles_sin.col(0), radiuspairs.col(0), pixel_a_y_imgcoord);
		pixel_a_x_imgcoord *= (m_Params->__max_raio_radius[stage] * m_TrainData->__data[index].__intermediate_bboxes[stage].width);
		pixel_a_y_imgcoord *= (m_Params->__max_raio_radius[stage] * m_TrainData->__data[index].__intermediate_bboxes[stage].height);

		cv::multiply(angles_cos.col(1), radiuspairs.col(1), pixel_b_x_imgcoord);
		cv::multiply(angles_sin.col(1), radiuspairs.col(1), pixel_b_y_imgcoord);
		pixel_b_x_imgcoord *= (m_Params->__max_raio_radius[stage] * m_TrainData->__data[index].__intermediate_bboxes[stage].width);
		pixel_b_y_imgcoord *= (m_Params->__max_raio_radius[stage] * m_TrainData->__data[index].__intermediate_bboxes[stage].height);

		//pixel_a_x_imgcoord = pixel_a_x_imgcoord * m_TrainData->__data[index].__meanshape2tf.at<double>(0, 0) + pixel_a_y_imgcoord * m_TrainData->__data[index].__meanshape2tf.at<double>(0, 1) + m_TrainData->__data[index].__meanshape2tf.at<double>(0, 2);
		//pixel_a_y_imgcoord = pixel_a_x_imgcoord * m_TrainData->__data[index].__meanshape2tf.at<double>(1, 0) + pixel_a_y_imgcoord * m_TrainData->__data[index].__meanshape2tf.at<double>(1, 1) + m_TrainData->__data[index].__meanshape2tf.at<double>(1, 2);

		//pixel_b_x_imgcoord = pixel_b_x_imgcoord * m_TrainData->__data[index].__meanshape2tf.at<double>(0, 0) + pixel_b_y_imgcoord * m_TrainData->__data[index].__meanshape2tf.at<double>(0, 1) + m_TrainData->__data[index].__meanshape2tf.at<double>(0, 2);
		//pixel_a_y_imgcoord = pixel_b_x_imgcoord * m_TrainData->__data[index].__meanshape2tf.at<double>(1, 0) + pixel_b_y_imgcoord * m_TrainData->__data[index].__meanshape2tf.at<double>(1, 1) + m_TrainData->__data[index].__meanshape2tf.at<double>(1, 2);

		cv::Mat_<double> pixel_a_x, pixel_a_y, pixel_b_x, pixel_b_y;

		pixel_a_x = pixel_a_x_imgcoord + m_TrainData->__data[index].__intermediate_shapes[stage](lmarkID, 0);
		pixel_a_y = pixel_a_y_imgcoord + m_TrainData->__data[index].__intermediate_shapes[stage](lmarkID, 1);

		pixel_b_x = pixel_b_x_imgcoord + m_TrainData->__data[index].__intermediate_shapes[stage](lmarkID, 0);
		pixel_b_y = pixel_b_y_imgcoord + m_TrainData->__data[index].__intermediate_shapes[stage](lmarkID, 1);

		int width = m_TrainData->__data[index].__width;
		int height = m_TrainData->__data[index].__height;

		for (int k = 0; k < pdfeats.rows; k++){
			int a_x = (int)(pixel_a_x(k, 0) + 0.5);
			int a_y = (int)(pixel_a_y(k, 0) + 0.5);
			int b_x = (int)(pixel_b_x(k, 0) + 0.5);
			int b_y = (int)(pixel_b_y(k, 0) + 0.5);

			a_x = MAX(0, MIN(a_x, width - 1));
			a_y = MAX(0, MIN(a_y, height - 1));
			b_x = MAX(0, MIN(b_x, width - 1));
			b_y = MAX(0, MIN(b_y, height - 1));

			double pixel_v_a = m_TrainData->__data[index].__img_gray(cv::Point(a_x, a_y));			
			double pixel_v_b = m_TrainData->__data[index].__img_gray(cv::Point(b_x, b_y));			

			pdfeats(k, j) = pixel_v_a - pixel_v_b;

		}
		shapes_residual(j, 0) = m_TrainData->__data[index].__shapes_residual(lmarkID, 0);
		shapes_residual(j, 1) = m_TrainData->__data[index].__shapes_residual(lmarkID, 1);		
	}	
	
	double E_x_2, E_x, E_y_2, E_y, var_overall;

	cv::Mat_<double> e_x_2;
	cv::pow(shapes_residual.col(0), 2.0, e_x_2);

	E_x_2 = cv::mean(e_x_2).val[0];
	E_x = cv::mean(shapes_residual.col(0)).val[0];

	cv::Mat_<double> e_y_2;
	cv::pow(shapes_residual.col(1), 2.0, e_y_2);

	E_y_2 = cv::mean(e_y_2).val[0];
	E_y = cv::mean(shapes_residual.col(1)).val[0];

	var_overall = length *((E_x_2 - E_x * E_x) + (E_y_2 - E_y * E_y));	

	cv::Mat_<double> var_reductions = cv::Mat::zeros(m_Params->__max_numfeats[stage], 1, CV_64FC1);
	cv::Mat_<double> thresholds = cv::Mat::zeros(m_Params->__max_numfeats[stage], 1, CV_64FC1);
	cv::Mat_<double> pdfeats_sorted;

	cv::sort(pdfeats, pdfeats_sorted, CV_SORT_EVERY_ROW | CV_SORT_ASCENDING);	

	for (int i = 0; i < m_Params->__max_numfeats[stage]; i++){
		int  ind = (int)floor(length * (0.5 + 0.9 * (__rand_01() - 0.5)));

		double threshold = pdfeats_sorted(i, ind);
		thresholds(i, 0) = threshold;

		cv::Mat_<int> ind_lc = cv::Mat::zeros(1, pdfeats.cols, CV_32SC1);
		cv::Mat_<int> ind_rc = cv::Mat::zeros(1, pdfeats.cols, CV_32SC1);

		cv::Mat_<double> shapes_residual_tmp;
		for (int j = 0; j < pdfeats.cols; j++){
			if (pdfeats(i, j) < threshold){
				ind_lc(0, j) = 1;			
				shapes_residual_tmp.push_back(shapes_residual.row(j));
			}
			else{
				ind_rc(0, j) = 1;
			}
		}
	
		if (!shapes_residual_tmp.empty()){
			cv::Mat_<double> e_x_2_lc;
			cv::pow(shapes_residual_tmp.col(0), 2.0, e_x_2_lc);
			double E_x_2_lc = cv::mean(e_x_2_lc).val[0];
			double E_x_lc = cv::mean(shapes_residual_tmp.col(0)).val[0];

			cv::Mat_<double> e_y_2_lc;
			cv::pow(shapes_residual_tmp.col(1), 2.0, e_y_2_lc);
			double E_y_2_lc = cv::mean(e_y_2_lc).val[0];
			double E_y_lc = cv::mean(shapes_residual_tmp.col(1)).val[0];

			double var_lc = (E_x_2_lc + E_y_2_lc) - (E_x_lc * E_x_lc + E_y_lc * E_y_lc);

			double sum1 = cv::sum(ind_lc).val[0];
			double sum2 = cv::sum(ind_rc).val[0];

			double E_x_2_rc = (E_x_2 * length - E_x_2_lc * sum1) / sum2;
			double E_x_rc = (E_x * length - E_x_lc*sum1) / sum2;

			double E_y_2_rc = (E_y_2 * length - E_y_2_lc*sum1) / sum2;
			double E_y_rc = (E_y * length - E_y_lc*sum1) / sum2;

			double var_rc = (E_x_2_rc + E_y_2_rc) - (E_x_rc * E_x_rc + E_y_rc * E_y_rc);
			double var_reduce = var_overall - sum1 * var_lc - sum2 * var_rc;

			var_reductions(i, 0) = var_reduce;
		}
		else{
			var_reductions(i, 0) = DBL_MIN;
		}

	}

	cv::Point maxloc;
	cv::minMaxLoc(var_reductions, NULL, NULL, NULL, &maxloc);
	int ind_colmax = maxloc.y;
	rt->__thresh = thresholds(ind_colmax, 0);

	rt->__feat = cv::Mat::zeros(1, 4, CV_64FC1);
	rt->__feat(0, 0) = anglepairs(ind_colmax, 0);
	rt->__feat(0, 1) = anglepairs(ind_colmax, 1);
	rt->__feat(0, 2) = radiuspairs(ind_colmax, 0);
	rt->__feat(0, 3) = radiuspairs(ind_colmax, 1);	

	for (int i = 0; i < pdfeats.cols; i++){
		cv::Mat_<int> equal = cv::Mat::zeros(1, 1, CV_32SC1);
		equal(0, 0) = ind_samples(0, i);
		if (pdfeats(ind_colmax, i) < rt->__thresh){
			rt->__lcind.push_back(equal);
		}
		else{
			rt->__rcind.push_back(equal);
		}
	}	

	if (!rt->__lcind.empty()){
		rt->__lcind = rt->__lcind.t();
	}
	if (!rt->__rcind.empty()){
		rt->__rcind = rt->__rcind.t();
	}		
}

double cRandomForest::__rand_01()
{
	boost::mt19937 engine;
	boost::random_device rd;
	engine.seed(rd());
	boost::uniform_01<boost::mt19937&> u01(engine);
	return u01();
}

void cRandomForest::__getproposals(int num_proposals, cv::Mat_<double>& radiuspairs, cv::Mat_<double>& anglepairs)
{
	int num_radius = sizeof(g_radius) / sizeof(g_radius[0]);
	int num_angles = sizeof(g_angles) / sizeof(g_angles[0]);

	cv::Mat_<int> tmp1 = __randperm(num_radius * num_angles);
	cv::Mat_<int> tmp2 = __randperm(num_radius * num_angles);

	cv::Mat_<int> id_radius_a, id_radius_b, Pro_a, Pro_b, Pro_a_choose, Pro_b_choose;

	for (int i = 0; i < num_radius * num_angles; i++){
		if (tmp1(0, i) != tmp2(0,i)){
			Pro_a.push_back(tmp1.col(i));
			Pro_b.push_back(tmp2.col(i));
		}
	}

	Pro_a = Pro_a.t();
	Pro_b = Pro_b.t();

	Pro_a_choose = Pro_a.colRange(0, num_proposals);
	Pro_b_choose = Pro_b.colRange(0, num_proposals);

	radiuspairs = cv::Mat::zeros(num_proposals, 2, CV_64FC1);
	anglepairs  = cv::Mat::zeros(num_proposals, 2, CV_64FC1);

	for (int i = 0; i < num_proposals; i++){
		int id_a = Pro_a_choose(0, i);
		int id_b = Pro_b_choose(0, i);

		int id_angles_a = id_a % num_angles;
		int id_angles_b = id_b % num_angles;

		int id_radius_a = (int)floor(float(id_a - 1) / num_angles);
		int id_radius_b = (int)floor(float(id_b - 1) / num_angles);

		radiuspairs(i, 0) = g_radius[id_radius_a];
		radiuspairs(i, 1) = g_radius[id_radius_b];

		anglepairs(i, 0) = g_angles[id_angles_a];
		anglepairs(i, 1) = g_angles[id_angles_b];
	}	
}

cv::Mat_<int> cRandomForest::__randperm(int permutation)
{
	std::vector<int> rng;
	for (int i = 0; i < permutation; i++){
		rng.push_back(i);
	}
	boost::random_shuffle(rng);

	cv::Mat_<int> res = cv::Mat::zeros(1, permutation, CV_32SC1);

	for (int i = 0; i < permutation; i++){
		res(0, i) = rng[i];
	}
	return res;
}