// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include <sstream>
#include <iostream>

#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/assign.hpp>
#include <boost/format.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/signals2.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <tbb/tbb.h>

#include <ppl.h>
#include <concurrent_vector.h>

#ifdef _DEBUG
#pragma comment(lib, "opencv_core232d")
#pragma comment(lib, "opencv_imgproc232d")
#pragma comment(lib, "opencv_features2d232d")
#pragma comment(lib, "opencv_ml232d")
#pragma comment(lib, "opencv_gpu232d")
#else
#pragma comment(lib, "opencv_core232")
#pragma comment(lib, "opencv_imgproc232")
#pragma comment(lib, "opencv_features2d232")
#pragma comment(lib, "opencv_ml232")
#pragma comment(lib, "opencv_gpu232")
#endif


const double SQRT_2 = std::sqrt(2.0);
const double PI = CV_PI;
