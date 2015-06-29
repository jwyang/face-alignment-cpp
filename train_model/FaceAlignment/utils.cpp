#include "utils.h"
#include <opencv2/video/tracking.hpp>

cv::Mat_<double>  cUtils::Calc_Point_Transform(const cv::Mat_<double>&src, const cv::Mat_<double>&dst)
{
	std::vector<cv::Point2f> src_pt, dst_pt;
	cv::Point2f pt_src, pt_dst;

	for (int i = 0; i < src.rows; i++){
		pt_src.x = (float)src(i, 0);
		pt_src.y = (float)src(i, 1);
		src_pt.push_back(pt_src);

		pt_dst.x = (float)dst(i, 0);
		pt_dst.y = (float)dst(i, 1);
		dst_pt.push_back(pt_dst);
	}
	cv::Mat H = cv::estimateRigidTransform(src_pt, dst_pt, false);

	if (H.cols * H.rows == 0){
		H = cv::Mat(2, 3, CV_64FC1);
		H.at<double>(0, 0) = 1.0;
		H.at<double>(0, 1) = 0.0;
		H.at<double>(0, 2) = 0.0;
		H.at<double>(1, 0) = 0.0;
		H.at<double>(1, 1) = 1.0;
		H.at<double>(1, 2) = 0.0;
	}

	return H;
}

cv::Mat_<double>   cUtils::Reshape(const cv::Mat& modelmean, cv::Rect& faceBox, sParams *params)
{
	cv::Mat_<double> modelShape = modelmean.clone();
	cv::Mat_<double> xCoords = modelShape.colRange(0, modelShape.cols / 2);
	cv::Mat_<double> yCoords = modelShape.colRange(modelShape.cols / 2, modelShape.cols);

	double minX, maxX, minY, maxY;
	cv::minMaxLoc(xCoords, &minX, &maxX);//得到x的最大/最小值
	cv::minMaxLoc(yCoords, &minY, &maxY);//得到y的最大/最小值
	double faceboxScaleFactor = params->__facebox_scale_factor;
	double modelWidth = maxX - minX;
	double modelHeight = maxY - minY;

	// scale it:
	modelShape = modelShape * (faceBox.width / modelWidth + faceBox.height / modelHeight) / (params->__facebox_scale_const * faceboxScaleFactor);
	// translate the model:
	cv::Scalar meanX = cv::mean(xCoords);
	double meanXd = meanX[0];
	cv::Scalar meanY = cv::mean(yCoords);
	double meanYd = meanY[0];
	// move it:
	xCoords += faceBox.x + faceBox.width / params->__facebox_width_div - meanXd;
	yCoords += faceBox.y + faceBox.height / params->__facebox_height_div - meanYd;
	return modelShape;
}

cv::Mat_<double>   cUtils::Reshape_alt(const cv::Mat& modelmean, cv::Rect& faceBox, sParams *params)
{
	cv::Mat_<double> modelShape = modelmean.clone();
	cv::Mat_<double> xCoords = modelShape.colRange(0, modelShape.cols / 2);
	cv::Mat_<double> yCoords = modelShape.colRange(modelShape.cols / 2, modelShape.cols);

	double minX, maxX, minY, maxY;
	cv::minMaxLoc(xCoords, &minX, &maxX);//得到x的最大/最小值
	cv::minMaxLoc(yCoords, &minY, &maxY);//得到y的最大/最小值
	double faceboxScaleFactor = params->__facebox_scale_factor;
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