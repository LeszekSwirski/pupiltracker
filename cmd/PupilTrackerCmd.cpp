#include <iostream>

#include <opencv2/highgui/highgui.hpp>

#include "../lib/PupilTracker.h"
#include "../lib/cvx.h"

void imshowscale(const std::string& name, cv::Mat& m, double scale)
{
    cv::Mat res;
    cv::resize(m, res, cv::Size(), scale, scale, cv::INTER_NEAREST);
    cv::imshow(name, res);
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Need filename" << std::endl;
        return 1;
    }

    std::cout << "Opening " << argv[1] << std::endl;
    cv::VideoCapture vc(argv[1]);
    if (!vc.isOpened()) {
        std::cerr << "Could not open " << argv[1] << std::endl;
        return 2;
    }


    cv::Mat m;
    while (true)
    {
        vc >> m;
        if (m.empty())
        {
            vc.open(argv[1]);
            if (!vc.isOpened()) {
                std::cerr << "Could not open " << argv[1] << std::endl;
                return 2;
            }
            vc >> m;
            if (m.empty()) {
                return 1;
            }
        }


        pupiltracker::TrackerParams params;
		params.Radius_Min = 3;
		params.Radius_Max = 8;

		params.CannyBlur = 1.6;
		params.CannyThreshold1 = 30;
		params.CannyThreshold2 = 50;
		params.StarburstPoints = 0;

		params.PercentageInliers = 40;
		params.InlierIterations = 2;
		params.ImageAwareSupport = true;
		params.EarlyTerminationPercentage = 95;
		params.EarlyRejection = true;
		params.Seed = -1;

        pupiltracker::findPupilEllipse_out out;
        pupiltracker::tracker_log log;
        pupiltracker::findPupilEllipse(params, m, out, log); 

        pupiltracker::cvx::cross(m, out.pPupil, 5, pupiltracker::cvx::rgb(255,255,0));
        cv::ellipse(m, out.elPupil, pupiltracker::cvx::rgb(255,0,255));


        imshowscale("Haar Pupil", out.mHaarPupil, 3);
        imshowscale("Pupil", out.mPupil, 3);
        //imshowscale("Thresh Pupil", out.mPupilThresh, 3);
        imshowscale("Edges", out.mPupilEdges, 3);
        cv::imshow("Result", m);

        if (cv::waitKey(10) != -1)
            break;
    }
}
