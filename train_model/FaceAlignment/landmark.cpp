#include "landmark.h"
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

cIBug::cIBug(sParams *params)
{	
	m_TrainData.__dbsizes = 0;

	m_Params = params;
	m_Pictures  = 0;	
	m_TrainData.__numlandmarks = 0;
	m_Identity = NULL;
}

cIBug::~cIBug()
{
	if (m_Identity) delete m_Identity;	
}

void cIBug::GetMaterial()
{	
	__init();
	__load();

	if (m_Params->__isflip) {
		__flip();
	}
}

void  cIBug::__init()
{
	int init_number = 0;
	if (m_Params->__landmark_type == std::string("LANDMARK_TYPE_51")){
		m_TrainData.__numlandmarks = 51;
		m_Identity = new int[51];
		init_number = 18;
		
		for (int i = 0; i < 51; i++){
			m_Identity[i] = init_number;
			init_number++;
		}
	}
	else if (m_Params->__landmark_type == std::string("LANDMARK_TYPE_68")){
		m_TrainData.__numlandmarks = 68;
		m_TrainData.__numlandmarks = 68;
		m_Identity = new int[68];
		init_number = 1;
		
		for (int i = 0; i < 68; i++){
			m_Identity[i] = init_number;
			init_number++;
		}
	}
}

void cIBug::__load()
{
	std::vector<std::string> paths_trainset(0);

	for (int i = 0; i < m_Params->__images_path.size();i++){
		try{
			std::ifstream fin(m_Params->__images_path[i]);
			if (fin.is_open()){
				while (!fin.eof()){
					std::string image_name;
					std::getline(fin, image_name);
					if (!image_name.empty()){
						paths_trainset.push_back(image_name);
					}
				}
			}
			fin.close();
		}catch (std::exception &e){
			std::cout << e.what() << std::endl;
		}
	}

	std::cout << "training set: " << paths_trainset.size() << std::endl;
	__parase(paths_trainset);
}

void cIBug::__parase(const std::vector<std::string>& paths)
{
	m_TrainData.__data.resize(paths.size());
// #pragma omp parallel for
	for (int ns = 0; ns < paths.size(); ++ns) {
		//try{
			//be sure 
			std::vector<std::string> img_split;
			boost::split(img_split, paths[ns], boost::is_any_of("."));
			std::string landname = img_split[0] + std::string(".pts");
			if (!boost::filesystem::exists(paths[ns]) || !boost::filesystem::exists(landname)){
				continue;
			}

			//read images
			cv::Mat_<uchar> img = cv::imread(paths[ns], 0);

			//read groundtruths
			std::ifstream fin(landname);
			std::string line;
			std::vector<std::string>container;
			std::getline(fin, line);
			std::getline(fin, line);
			boost::split(container, line, boost::is_any_of(" "));
			int landmarks = boost::lexical_cast<int>(container[2]);
			cv::Mat_<double> land = cv::Mat::zeros(landmarks, 2, CV_64FC1);

			//get all
			int x_tl(RAND_MAX), y_tl(RAND_MAX), x_br(0), y_br(0);
			std::getline(fin, line);
			for (int i = 0; i < landmarks; i++){
				std::getline(fin, line);
				boost::split(container, line, boost::is_any_of(" "));
				land(i, 0) = (boost::lexical_cast<double>(container[0]) - 1.0);
				land(i, 1) = (boost::lexical_cast<double>(container[1]) - 1.0);
				x_tl = MIN(x_tl, land(i, 0));
				y_tl = MIN(y_tl, land(i, 1));
				x_br = MAX(x_br, land(i, 0));
				y_br = MAX(y_br, land(i, 1));
			}
			fin.close();

			cv::Rect bbox_land = cv::Rect(x_tl, y_tl, x_br - x_tl + 1, y_br - y_tl + 1);

			std::vector<cv::Rect> detectedFaces(0);
			m_Params->__facecascade.detectMultiScale(img, detectedFaces, 1.2, 2, 0, cv::Size(50, 50));
			if (detectedFaces.size() == 0) {
				detectedFaces.push_back(bbox_land);
			}
			else {
				bool is_valid = false;
				for (int i = 0; i < detectedFaces.size(); i++){
					cv::Rect bbox = detectedFaces[i];
					if (bbox.contains(cv::Point((int)land(36, 0), (int)land(36, 1))) && bbox.contains(cv::Point((int)land(45, 0), (int)land(45, 1)))
						&& bbox.contains(cv::Point((int)land(57, 0), (int)land(57, 1)))){
						is_valid = true;
						break;
					}
				}
				if (!is_valid) {
					detectedFaces.clear();
					detectedFaces.push_back(bbox_land);
				}
			}

			//cv::rectangle(img,, CV_RGB(255, 0, 0), 2);
			//cv::imshow("img", img);
			//cv::waitKey(0);


			for (int i = 0; i < detectedFaces.size(); i++){
				cv::Rect bbox = detectedFaces[i];
				if (bbox.contains(cv::Point((int)land(36, 0), (int)land(36, 1))) && bbox.contains(cv::Point((int)land(45, 0), (int)land(45, 1)))
					&& bbox.contains(cv::Point((int)land(57, 0), (int)land(57, 1)))){

					/* to save memory, we crop a roi from original image which is 2 times of face bounding box*/
					cv::Rect roi;
					__clip(bbox, roi, land, img.cols, img.rows, 2.0);
					cv::Mat img_roi(img(roi));

					// resize materials
					double ratio = MAX(1.0, double(bbox.width * bbox.height) / (200 * 200));

					cv::resize(img_roi, img_roi, cv::Size(int(img_roi.cols / ratio), int(img_roi.rows / ratio)));
					bbox = cv::Rect(bbox.x / ratio, bbox.y / ratio, bbox.width / ratio, bbox.height / ratio);
					// cv::rectangle(img_roi, bbox, 255, 2);
					m_TrainData.__data[ns].__width = img_roi.cols;
					m_TrainData.__data[ns].__height = img_roi.rows;
					m_TrainData.__data[ns].__bbox_gt = bbox;

					//get a part of points
					cv::Mat_<double> groundtruth = cv::Mat::zeros(m_TrainData.__numlandmarks, 2, CV_64FC1);
					for (int j = 0; j < m_TrainData.__numlandmarks; j++){
						int land_x = (int)(land(m_Identity[j] - 1, 0) + 0.5);
						int land_y = (int)(land(m_Identity[j] - 1, 1) + 0.5);
						groundtruth(j, 0) = (double)land_x / ratio;
						groundtruth(j, 1) = (double)land_y / ratio;
						// cv::circle(img_roi, cv::Point(groundtruth(j, 0), groundtruth(j, 1)), 2, 255);
					}

					groundtruth.copyTo(m_TrainData.__data[ns].__shape_gt);
					// std::cout << "width: [" << roi.width << "], height: [" << roi.height << "]" << std::endl;
					// cv::imshow("roi", img_roi);
					// cv::waitKey(0);

					img_roi.copyTo(m_TrainData.__data[ns].__img_gray);

					m_TrainData.__data[ns].__intermediate_shapes.resize(m_Params->__max_numstage + 1);
					m_TrainData.__data[ns].__intermediate_bboxes.resize(m_Params->__max_numstage + 1);

					break;
					// if (m_Pictures % 100 == 0)
					// std::cout << "Loaded [" << m_Pictures << "] pictures" << std::endl;			
				}//end if(bbox)
				
			}//end for		
		//}
		//catch (std::exception &e){
			//std::cout << e.what() << std::endl;
		//}
	}

	m_Pictures = paths.size();
	m_TrainData.__dbsizes = m_Pictures;
}

cv::Rect cIBug::__enlargebox(const cv::Rect &rect)
{
	cv::Rect region;
	region.x = (int)floor(rect.x - 0.5 * rect.width);
	region.y = (int)floor(rect.y - 0.5 * rect.height);
	region.width = (int)floor(float(2 * rect.width));
	region.height = (int)floor(float(2 * rect.height));
	return region;
}

void cIBug::__clip(cv::Rect& box, cv::Rect& roi, cv::Mat_<double> &land, int width, int height, double scale)
{
	roi = __enlargebox(box); // enlarge box to be 2 times of origin one

	// regularize box
	roi.x = MAX(roi.x, 0);
	roi.y = MAX(roi.y, 0);
	int right_x  = MIN(roi.width  + roi.x, width);
	int bottom_y = MIN(roi.height + roi.y, height);

	roi.width  = right_x  - roi.x;
	roi.height = bottom_y - roi.y;

	land.col(0) -= (double)roi.x;
	land.col(1) -= (double)roi.y;

	// shift face region accordingly
	box.x -= roi.x;
	box.y -= roi.y;
}

void cIBug::__flip()
{
// #pragma omp parallel for 
	for (int i = 0; i < m_Pictures; i++){
		cv::Mat_<uchar> img;
		// flip image
		cv::flip(m_TrainData.__data[i].__img_gray, img, 1);

		// flip shape
		cv::Mat_<double> land = m_TrainData.__data[i].__shape_gt.clone();
		__flipshape(land, img.cols);
		/*
		for (int j = 0; j < m_TrainData.__numlandmarks; ++j) {
			cv::circle(img, cv::Point(land(j, 0), land(j, 1)), 2, 255);
			cv::imshow("flip", img);
			cv::waitKey(0);
		}
		*/
		// flip bbox
		cv::Rect bbox_flip = m_TrainData.__data[i].__bbox_gt;
		bbox_flip.x = img.cols - bbox_flip.x - bbox_flip.width;

		/*
		cv::rectangle(img, bbox_flip, 255, 2);
		cv::imshow("flip", img);
		cv::waitKey(0);
		*/

		sData data;

		data.__width = img.cols;
		data.__height = img.rows;
		data.__bbox_gt = bbox_flip;

		data.__shape_gt = land.clone();
		data.__img_gray = img.clone();

		data.__intermediate_shapes.resize(m_Params->__max_numstage + 1);
		data.__intermediate_bboxes.resize(m_Params->__max_numstage + 1);

		m_TrainData.__data.push_back(data);
	}//end for (int i=0;)
	m_TrainData.__dbsizes += m_Pictures;
}

void cIBug::__flipshape(cv::Mat_<double> &land, int width)
{
	int p = 0, q = 0;
	if (m_TrainData.__numlandmarks == 51){
		for (p = 0, q = 9; p < q; p++,q--){
			__exchange(land, p, q);
		}

		for (p = 14, q = 18; p < q; p++, q--){
			__exchange(land, p, q);
		}

		__exchange(land, 19, 28);
		__exchange(land, 20, 27);
		__exchange(land, 21, 26);
		__exchange(land, 22, 25);
		__exchange(land, 23, 30);
		__exchange(land, 24, 29);

		for (p = 31, q = 37; p < q; p++, q--){
			__exchange(land, p, q);
		}

		for (p = 38, q = 42; p < q; p++, q--){
			__exchange(land, p, q);
		}

		for (p = 43, q = 47; p < q; p++, q--){
			__exchange(land, p, q);
		}

		for (p = 48, q = 50; p < q; p++, q--){
			__exchange(land, p, q);
		}

		for (int i = 0; i < land.rows; i++){
			land(i, 0) = width - land(i, 0);
		}

	}
	else if (m_TrainData.__numlandmarks == 68) {

		// flip 1-17
		for (p = 0, q = 16; p < q; p++, q--){
			__exchange(land, p, q);
		}

		// flip eyebow
		for (p = 17, q = 26; p < q; p++, q--){
			__exchange(land, p, q);
		}

		// flip left eye
		for (p = 31, q = 35; p < q; p++, q--){
			__exchange(land, p, q);
		}


		__exchange(land, 36, 45);
		__exchange(land, 37, 44);
		__exchange(land, 38, 43);
		__exchange(land, 39, 42);

		__exchange(land, 40, 47);
		__exchange(land, 41, 46);


		for (p = 48, q = 54; p < q; p++, q--){
			__exchange(land, p, q);
		}
		for (p = 55, q = 59; p < q; p++, q--){
			__exchange(land, p, q);
		}
		for (p = 60, q = 64; p < q; p++, q--){
			__exchange(land, p, q);
		}
		for (p = 65, q = 67; p < q; p++, q--){
			__exchange(land, p, q);
		}
		for (int i = 0; i < land.rows; i++){
			land(i, 0) = width - land(i, 0);
		}
	}
}

void cIBug::__exchange(cv::Mat_<double> &land, int index1, int index2)
{
	double tmp_x = land(index1, 0);
	double tmp_y = land(index1, 1);

	land(index1, 0) = land(index2, 0);
	land(index1, 1) = land(index2, 1);

	land(index2, 0) = tmp_x;
	land(index2, 1) = tmp_y;
}
