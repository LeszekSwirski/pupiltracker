#ifndef __CVX_H__
#define __CVX_H__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.h"

const double SQRT_2 = std::sqrt(2.0);
const double PI = CV_PI;

namespace cvx
{

	template<typename T>
	inline cv::Rect_<T> roiAround(T x, T y, T radius)
	{
		return cv::Rect_<T>(x - radius, y - radius, 2*radius + 1, 2*radius + 1);
	}
	template<typename T>
	inline cv::Rect_<T> roiAround(const cv::Point_<T>& centre, T radius)
	{
		return roiAround(centre.x, centre.y, radius);
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

	void getROI(const cv::Mat& src, cv::Mat& dst, const cv::Rect& roi, int borderType = cv::BORDER_REPLICATE);

	float histKmeans(const cv::Mat_<float>& hist, int bin_min, int bin_max, int K, float init_centres[], cv::Mat_<uchar>& labels, cv::TermCriteria termCriteria);

	cv::RotatedRect fitEllipse(const cv::Moments& m);
	cv::Vec2f majorAxis(const cv::RotatedRect& ellipse);




}

#endif // __CVX_H__
