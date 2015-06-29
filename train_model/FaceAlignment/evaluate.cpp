#include "evaluate.h"

cEvaluate::cEvaluate()
{
	return;
}


cEvaluate::~cEvaluate()
{
	return;
}

double cEvaluate::compute_error(std::vector<cv::Mat_<double>>& preshapes, std::vector<cv::Mat_<double>>& gtshapes) {
	assert(preshapes.size() == gtshapes.size());
	int _num_samples = preshapes.size();
	int _num_landmarks = preshapes[0].rows;

	double _error_average = 0;

	cv::Mat_<double> _error_samples(1, _num_samples);

	for (int i = 0; i < _num_samples; ++i) {
		double dist_interocular = 0;
		if (_num_landmarks == 68) {
			cv::Scalar mean_leye_x = cv::mean(gtshapes[i].rowRange(36, 41).col(0));
			cv::Scalar mean_leye_y = cv::mean(gtshapes[i].rowRange(36, 41).col(1));

			cv::Scalar mean_reye_x = cv::mean(gtshapes[i].rowRange(42, 47).col(0));
			cv::Scalar mean_reye_y = cv::mean(gtshapes[i].rowRange(42, 47).col(1));

			dist_interocular = sqrt(pow(mean_leye_x.val[0] - mean_reye_x.val[0], 2) + pow(mean_leye_y.val[0] - mean_reye_y.val[0], 2));
		}
		else if (_num_landmarks == 51) {
			;
		}
		else if (_num_landmarks == 29) {
			;
		}

		double norm_sum = 0;
		cv::Mat_<double> deltashape = preshapes[i] - gtshapes[i];

		for (int n = 0; n < _num_landmarks; ++n) {
			norm_sum += cv::norm(deltashape.row(n));
		}

		double norm_average = norm_sum / _num_landmarks;

		_error_samples(0, i) = norm_average / dist_interocular;
	}

	return 100 * cv::mean(_error_samples).val[0];
}
