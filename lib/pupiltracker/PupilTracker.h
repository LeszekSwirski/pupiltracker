#ifndef __PUPIL_TRACKER_PUPILTRACKER_H__
#define __PUPIL_TRACKER_PUPILTRACKER_H__

#include <vector>
#include <string>

#include <boost/lexical_cast.hpp>
#include <opencv2/core/core.hpp>

#include <pupiltracker/timer.h>
#include <pupiltracker/ConicSection.h>

namespace pupiltracker {

struct tracker_log
{
public:
    typedef std::vector< std::pair<std::string, std::string> >::iterator iterator;
    typedef std::vector< std::pair<std::string, std::string> >::const_iterator const_iterator;

    template<typename T>
    void add(const std::string& key, const T& val)
    {
        m_log.push_back(std::make_pair(key, boost::lexical_cast<std::string>(val)));
    }
    void add(const std::string& key, const timer& val)
    {
        std::stringstream ss;
        ss.precision(2);
        ss.setf(std::ios::fixed);
        ss << (val.elapsed()*1000.0) << "ms";
        m_log.push_back(std::make_pair(key, ss.str()));
    }

    iterator begin() { return m_log.begin(); }
    const_iterator begin() const { return m_log.begin(); }
    iterator end() { return m_log.end(); }
    const_iterator end() const { return m_log.end(); }

private:
    std::vector< std::pair<std::string, std::string> > m_log;
};
    
struct TrackerParams
{
    int Radius_Min;
    int Radius_Max;

    double CannyBlur;
    double CannyThreshold1;
    double CannyThreshold2;
    int StarburstPoints;

    int PercentageInliers;
    int InlierIterations;
    bool ImageAwareSupport;
    int EarlyTerminationPercentage;
    bool EarlyRejection;
    int Seed;
};

const cv::Point2f UNKNOWN_POSITION = cv::Point2f(-1,-1);

struct EdgePoint
{
    cv::Point2f point;
    double edgeStrength;

    EdgePoint(const cv::Point2f& p, double s) : point(p), edgeStrength(s) {}
    EdgePoint(float x, float y, double s) : point(x,y), edgeStrength(s) {}

    bool operator== (const EdgePoint& other)
    {
        return point == other.point;
    }
};

struct findPupilEllipse_out {
    cv::Rect roiHaarPupil;
    cv::Mat_<uchar> mHaarPupil;
    
    cv::Mat_<float> histPupil;
    double threshold;
    cv::Mat_<uchar> mPupilThresh;

    cv::Rect bbPupilThresh;
    cv::RotatedRect elPupilThresh;

    cv::Rect roiPupil;
    cv::Mat_<uchar> mPupil;
    cv::Mat_<uchar> mPupilOpened;
    cv::Mat_<uchar> mPupilBlurred;
    cv::Mat_<uchar> mPupilEdges;
    cv::Mat_<float> mPupilSobelX;
    cv::Mat_<float> mPupilSobelY;

    std::vector<EdgePoint> edgePoints;
    std::vector<cv::Point2f> inliers;
    int ransacIterations;
    int earlyRejections;
    bool earlyTermination;

    cv::Point2f pPupil;
    cv::RotatedRect elPupil;

    findPupilEllipse_out() : pPupil(UNKNOWN_POSITION), threshold(-1) {}
};
bool findPupilEllipse(
    const TrackerParams& params,
    const cv::Mat& m,

    findPupilEllipse_out& out,
    tracker_log& log
    );

} //namespace pupiltracker

#endif//__PUPIL_TRACKER_PUPILTRACKER_H__
