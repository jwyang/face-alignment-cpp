#ifndef PARAMS_H
#define PARAMS_H

#include <opencv2/objdetect/objdetect.hpp>
#include <string>
#include <vector>

typedef struct sParams{
	int     __max_numstage;
	int     __max_depth;
	int     __max_nodes;
	int*    __max_numfeats;
	int     __max_numtrees;
	int     __max_numthreshs;
	double  __bagging_overlap;
	double* __max_raio_radius;
	bool    __isflip;

	//mean face
	int    __procrustes_iters;
	double __procrustes_errors;
	double __facebox_scale_factor;
	double __facebox_scale_const;
	double __facebox_width_div;
	double __facebox_height_div;
	
	std::string               __landmark_type;
	std::string               __outputmodel;
	std::vector<std::string>  __images_path;
	cv::CascadeClassifier     __facecascade;
}sParams;

#endif