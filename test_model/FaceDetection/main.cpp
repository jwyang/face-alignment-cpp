#include "model.h"
#include "params.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

namespace po = boost::program_options;
using boost::filesystem::path;
using boost::property_tree::ptree;

int main(int argc, char *argv[])
{	
	path ModelFile;
	path faceDetectorFilename;
	path faceBoxesDirectory;
	path outputDirectory;
	path inputPaths;
	path tainPaths;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
			"Produce help message")			
			("input,i", po::value<path>(&inputPaths)->required(),
			"The path of image.")
			("model,m", po::value<path>(&ModelFile)->required(),
			"An  model file to load.")
			("train,t", po::value<path>(&tainPaths)->required(),
			"The train.txt.")
			("face-detector,f", po::value<path>(&faceDetectorFilename)->required(),
			"Path to an XML CascadeClassifier from OpenCV.")
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
		if (vm.count("face-detector")  != 1) {
			std::cout << "Error while parsing command-line arguments: specify either a face-detector (-f)  as input" << std::endl;
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

	try {
		read_info(tainPaths.string(), pt);
	}
	catch (const boost::property_tree::ptree_error& e) {
		std::cout << std::string("Error reading the config file: ") + e.what() << std::endl;
		return -EXIT_FAILURE;
	}

	//init mean face
	ptree meanface = pt.get_child("meanface");	
	params.__facebox_scale_factor = meanface.get<double>("SCALE_FACTOR");
	params.__facebox_scale_const = meanface.get<double>("SCALE_CONST");
	params.__facebox_width_div = meanface.get<double>("WIDTH_DIV");
	params.__facebox_height_div = meanface.get<double>("HEIGHT_DIV");

	cv::CascadeClassifier faceCascade;

	if (!faceCascade.load(faceDetectorFilename.string()))
	{
		std::cout << "Error loading the face detection model." << std::endl;
		return EXIT_FAILURE;
	}

	cv::Mat img = cv::imread(inputPaths.string(), 0);
	if (img.empty()){
		std::cout << "Error loading the image." << std::endl;
		return EXIT_FAILURE;
	}
	
	cv::Mat img_dis = cv::imread(inputPaths.string());

	std::vector<cv::Rect> detectedFaces;
	faceCascade.detectMultiScale(img, detectedFaces, 1.2, 2, 0, cv::Size(50, 50));

	if (!detectedFaces.empty()){
		cModel app(ModelFile.string(), &params);
		app.Init();
		sModel model = app.GetModel();
		int stages = model.__head.__num_stage;
		cv::Mat_<float> meanface = model.__meanface;		

		for (int i = 0; i < detectedFaces.size(); i++){	
			cv::Mat_<float> shape = app.Reshape_alt(meanface, detectedFaces[i]);			
			double t = (double)cvGetTickCount();
			for (int j = 0; j < stages; j++){
				cv::Mat_<int> binary = app.DerivBinaryfeat(img, detectedFaces[i], shape, j);
				app.UpDate(binary, detectedFaces[i], shape, j);
			}
			t = (double)cvGetTickCount() - t;
			std::cout << "Alignment runtime:" << t / (cvGetTickFrequency() * 1000) << " ms" << std::endl;

			for (int m = 0; m < shape.rows; m++){
				cv::circle(img_dis, cv::Point((int)shape(m, 0), (int)shape(m, 1)), 1, cv::Scalar(0, 255, 0));
			}
		}

		cv::imshow("reslut", img_dis);
		cv::waitKey();
	}else{
		std::cout << "No faces detect!." << std::endl;
	}

	return 0;
}