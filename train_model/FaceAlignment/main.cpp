#include "train.h"
#include <boost/program_options.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

#include <iostream>

namespace po = boost::program_options;
using boost::filesystem::path;
using boost::property_tree::ptree;

void Init_Params(sParams *params)
{
	params->__max_numfeats    = NULL;
	params->__max_raio_radius = NULL;
}

int main(int argc, char *argv[])
{
	path outputfile;
	path configfile;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
			"Produce help message")
			("input,i", po::value<path>(&configfile)->required(),
			"The path of config file.")
			("model,o", po::value<path>(&outputfile)->required(),
			"The path of output.")			
			;

		po::positional_options_description p;
		p.add("input", -1);

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
		if (vm.count("help")) {
			std::cout << "Usage: detect-landmarks [options]" << std::endl;
			std::cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
		if (vm.count("input") != 1) {
			std::cout << "Error while parsing command-line arguments: specify either a input(-f)  as input" << std::endl;
			std::cout << desc;
			return EXIT_SUCCESS;
		}
	}
	catch (po::error& e) {
		std::cout << "Error while parsing command-line arguments: " << e.what() << std::endl;
		std::cout << "Use --help to display a list of options." << std::endl;
		return EXIT_SUCCESS;
	}

	ptree pt;
	sParams params;	
	Init_Params(&params);

	try {
		read_info(configfile.string(), pt);
	}
	catch (const boost::property_tree::ptree_error& e) {
		std::cout << std::string("Error reading the config file: ") + e.what()<<std::endl;
		return -EXIT_FAILURE;
	}
	
	try{
		ptree parameters         = pt.get_child("parameters");
		params.__max_numstage    = parameters.get<int>("MAX_STAGES");
		params.__max_depth       = parameters.get<int>("MAX_DEPTH");
		params.__max_nodes       = (1 << params.__max_depth)-1;
		params.__max_numtrees    = parameters.get<int>("MAX_NUMTRESS");
		params.__max_numthreshs  = parameters.get<int>("MAX_NUMTHRESHS");
		params.__bagging_overlap = parameters.get<double>("BAGGING_OVERLAP");		
		params.__isflip          = parameters.get<bool>("IS_FLIP"); 
		params.__outputmodel     = outputfile.string();

		std::string str_numfeats = parameters.get<std::string>("MAX_NUMFEATS");
		std::string str_ration   = parameters.get<std::string>("MAX_RATIO_RADIUS");

		std::vector<std::string> split_numfeats;
		std::vector<std::string> split_ration;

		boost::split(split_numfeats, str_numfeats, boost::is_any_of(","));
		int num_numfeats = 0;
		for (int i = 0; i < split_numfeats.size(); i++){
			if (split_numfeats[i] != std::string(",")){
				num_numfeats++;
			}
		}
		params.__max_numfeats = new int[num_numfeats];
		int index = 0;
		for (int i = 0; i < split_numfeats.size(); i++){
			if (split_numfeats[i] != std::string(",")){
				params.__max_numfeats[index] = boost::lexical_cast<int>(split_numfeats[i]);
				index++;
			}
		}

		boost::split(split_ration, str_ration, boost::is_any_of(","));
		int num_ration = 0;
		for (int i = 0; i < split_ration.size(); i++){
			if (split_ration[i] != std::string(",")){
				num_ration++;
			}
		}
		params.__max_raio_radius = new double[num_ration];
		for (int i = 0, index = 0; i < split_ration.size(); i++){
			if (split_ration[i] != std::string(",")){
				params.__max_raio_radius[index] = boost::lexical_cast<double>(split_ration[i]);
				index++;
			}
		}
		//init mean face
		ptree meanface = pt.get_child("meanface");
		params.__procrustes_iters     = meanface.get<int>("MAX_ITERS");
		params.__procrustes_errors    = meanface.get<double>("MAX_ERRORS");
		params.__facebox_scale_factor = meanface.get<double>("SCALE_FACTOR");
		params.__facebox_scale_const  = meanface.get<double>("SCALE_CONST");
		params.__facebox_width_div    = meanface.get<double>("WIDTH_DIV");
		params.__facebox_height_div   = meanface.get<double>("HEIGHT_DIV");

		//init train
		ptree train = pt.get_child("train");
		std::string facedetect_file = train.get<std::string>("FACE_DETECT_PATH");

		if (!params.__facecascade.load(facedetect_file)){
			std::cout << "failed to load " << facedetect_file << std::endl;
			if (params.__max_numfeats){
				delete params.__max_numfeats;
			}
			return -EXIT_FAILURE;
		}
	
		params.__landmark_type = train.get<std::string>("LANDMARK_TYPE");
		
		ptree dataset = train.get_child("dataset");
		for (auto it = dataset.begin(); it != dataset.end(); it++){
			std::string dataset_str = it->second.get<std::string>("DATA_PATH");
			params.__images_path.push_back(dataset_str);
		}

	}
	catch (const boost::property_tree::ptree_error& error){
		std::cout << std::string("Parsing config: ") + error.what() << std::endl;
		return -EXIT_FAILURE;
	}

	cTrain app(&params);
	
	// try{
		app.Train();
	// }
	// catch(std::exception &e){
	//	std::cout << std::string("train model: ") + e.what() << std::endl;
	// }

	delete params.__max_numfeats;
	delete params.__max_raio_radius;
	
	return EXIT_SUCCESS;
}