#ifndef cvx_h__
#define cvx_h__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.h"

namespace cvx
{
	// Octave<->sigma conversion functions
	// Used Matlab to minimise difference between Gaussian of sigma 's'
	// versus pyramid image of level 'n'.
	// Got
	//   n |  0   1   2   3   4   5   6   7   8 
	//   s |  0
	// Modelled as:
	//   s = sqrt( (a^n - 1) / b )
	// where
	//   a = 3.9512
	//   b = 1.4182
	// and
	//   n = log_a(b * s^2 + 1)

	namespace
	{
		const double a = 3.9512;
		const double b = 1.4182;
		const double loga = std::log(a);
	}

	inline double octave2sigmaSq(int octave, double sigma_rem = 0.0)
	{
		return (std::pow(a, octave) - 1)/b + sq(sigma_rem);
	}
	inline double octave2sigma(int octave, double sigma_rem = 0.0)
	{
		return std::sqrt(octave2sigmaSq(octave, sigma_rem));
	}
	inline int sigma2octave(double sigma)
	{
		return (int)std::floor( std::log( b * sq(sigma) + 1) / loga );
	}
	inline int sigma2octave(double sigma, double& sigma_rem)
	{
		int octave = sigma2octave(sigma);
		sigma_rem = std::sqrt( sq(sigma) - octave2sigmaSq(octave));
		return octave;
	}


	template<typename T>
	inline cv::Rect_<T> roiAround(const cv::Point_<T>& centre, T radius)
	{
		return roiAround(centre.x, centre.y, radius);
	}
	template<typename T>
	inline cv::Rect_<T> roiAround(T x, T y, T radius)
	{
		return cv::Rect_<T>(x - radius, y - radius, 2*radius + 1, 2*radius + 1);
	}



	inline void cross(cv::Mat& img, cv::Point centre, int radius, const cv::Scalar& colour, int thickness = 1, int lineType = 8, int shift = 0)
	{
		cv::line(img, centre + cv::Point(-radius, -radius), centre + cv::Point(radius, radius), colour, thickness, lineType, shift = 0);
		cv::line(img, centre + cv::Point(-radius, radius), centre + cv::Point(radius, -radius), colour, thickness, lineType, shift = 0);
	}
	inline void plus(cv::Mat& img, cv::Point centre, int radius, const cv::Scalar& colour, int thickness = 1, int lineType = 8, int shift = 0)
	{
		cv::line(img, centre + cv::Point(0, -radius), centre + cv::Point(0, radius), colour, thickness, lineType, shift = 0);
		cv::line(img, centre + cv::Point(-radius, 0), centre + cv::Point(radius, 0), colour, thickness, lineType, shift = 0);
	}

	inline cv::Rect boundingBox(const cv::Mat& img)
	{
		return cv::Rect(0,0,img.cols,img.rows);
	}

	void blurAndDecimate(const cv::Mat& src, cv::Mat& dst, double sigma, int& octave, int borderType = cv::BORDER_REPLICATE);
	void getROI(const cv::Mat& src, cv::Mat& dst, const cv::Rect& roi, int borderType = cv::BORDER_REPLICATE);
	void hessianRatio(const cv::Mat& src, cv::Mat& dst);

	float histKmeans(const cv::Mat_<float>& hist, int bin_min, int bin_max, int K, float init_centres[], cv::Mat_<uchar>& labels, cv::TermCriteria termCriteria);

	cv::RotatedRect fitEllipse(const cv::Moments& m);
	cv::Vec2f majorAxis(const cv::RotatedRect& ellipse);




}

#endif // cvx_h__