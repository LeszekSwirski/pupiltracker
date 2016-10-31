#include <iostream>

#include <opencv2/highgui/highgui.hpp>

#include <pupiltracker/PupilTracker.h>
#include <pupiltracker/cvx.h>


//dlib libraries
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/threads.h>
#include <dlib/misc_api.h>

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/opencv.hpp"
#include <time.h>

#include <stdio.h>
#include <string>
#include <math.h>
#include <cmath>

using namespace dlib;
using namespace std;


const static cv::Point3f P3D_SELLION(0., 0.,0.);
const static cv::Point3f P3D_RIGHT_EYE(-20., -65.5,-5.);
const static cv::Point3f P3D_LEFT_EYE(-20., 65.5,-5.);
const static cv::Point3f P3D_RIGHT_EAR(-100., -77.5,-6.);
const static cv::Point3f P3D_LEFT_EAR(-100., 77.5,-6.);
const static cv::Point3f P3D_NOSE(21.0, 0., -48.0);
const static cv::Point3f P3D_STOMMION(10.0, 0., -75.0);
const static cv::Point3f P3D_MENTON(0., 0.,-133.0);

enum 
{
     NOSE=30,
     RIGHT_EYE=36,
     LEFT_EYE=45,
     RIGHT_SIDE=0,
     LEFT_SIDE=16,
     EYEBROW_RIGHT=21,
     EYEBROW_LEFT=22,
     MOUTH_UP=51,
     MOUTH_DOWN=57,
     MOUTH_RIGHT=48,
     MOUTH_LEFT=54,
     SELLION=27,
     MOUTH_CENTER_TOP=62,
     MOUTH_CENTER_BOTTOM=66,
     MENTON=8
};

string filename = "../../camera.yml";

int main(int argc, char* argv[])
{
   if(argc <= 1) 
   {
     
    std::cerr<<"Use ./DisplayImage dev_port_no boad_size"<<endl;
    return 1;
  }

    int video_source;
    std::istringstream video_sourceCmd(argv[1]);
     // Check if it is indeed a number
     if(!(video_sourceCmd >> video_source)){ cout<<"wrong video port no.!!"<<endl;return 1;}  
    
     cv::VideoCapture cap(video_source);

     cv::FileStorage fs;
    fs.open(filename, cv::FileStorage::READ);
    // read camera matrix and distortion coefficients from file
    cv::Mat intrinsics, distortion;
    fs["Camera_Matrix"] >> intrinsics;
    fs["Distortion_Coefficients"] >> distortion;
    cout<<" dis"<<intrinsics.at<double>(2, 3)<<endl;    
    // close the input file
    fs.release();




    std::vector<cv::Point3f> head_points;
    cv::Mat rvec = cv::Mat(cv::Size(3,1), CV_64F);
    CvMat r;
    cv::Mat tvec = cv::Mat(cv::Size(3,1), CV_64F);

    

    head_points.push_back(P3D_SELLION);
    head_points.push_back(P3D_RIGHT_EYE);
    head_points.push_back(P3D_LEFT_EYE);
    head_points.push_back(P3D_RIGHT_EAR);
    head_points.push_back(P3D_LEFT_EAR);
    head_points.push_back(P3D_MENTON);
    head_points.push_back(P3D_NOSE);
    head_points.push_back(P3D_STOMMION);




int n=0,m=0;

    try
    {

        cap.set( CV_CAP_PROP_FRAME_WIDTH, 640 );
        cap.set( CV_CAP_PROP_FRAME_HEIGHT, 480 );        
        //image_window win;
        cv::Mat left,right;
        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("../../shape_predictor_68_face_landmarks.dat") >> pose_model;



        pupiltracker::TrackerParams params;
        params.Radius_Min = 8;
        params.Radius_Max = 25;

        params.CannyBlur = 1;
        params.CannyThreshold1 = 20;
        params.CannyThreshold2 = 40;
        params.StarburstPoints = 0;

        params.PercentageInliers = 30;
        params.InlierIterations = 2;
        params.ImageAwareSupport = true;
        params.EarlyTerminationPercentage = 95;
        params.EarlyRejection = true;
        params.Seed = -1;

        pupiltracker::findPupilEllipse_out out,out1;
        pupiltracker::tracker_log log,log1;

        

        std::vector<cv::Point2f> imageFramePoints;
        std::vector<cv::Point3f> framePoints;

        framePoints.push_back(cv::Point3d(0.0,0.0,0.0));
        framePoints.push_back(cv::Point3d(50,0,0));
        framePoints.push_back(cv::Point3d(0,50,0));
        framePoints.push_back(cv::Point3d(0,0,50));

        // Grab and process frames until the main window is closed by the user.
        int count=0;
        int _lx=0,_ly=0,_rx=0,_ry=0,rX,rY,lX,lY;

        int _nosex=0,_nosey=0,nosex,nosey;
        int _right_eyex=0,_right_eyey=0,right_eyex,right_eyey;
        int _left_eyex=0,_left_eyey=0,left_eyex,left_eyey;
        int _right_sidex=0,_right_sidey=0,right_sidex,right_sidey;
        int _left_sidex=0,_left_sidey=0,left_sidex,left_sidey;
        int _mouth_rightx=0,_mouth_righty=0,mouth_rightx,mouth_righty;
        int _mouth_leftx=0, _mouth_lefty=0,mouth_leftx, mouth_lefty;  
        int _mouth_downx=0,_mouth_downy=0,mouth_downx,mouth_downy;
        int _mouth_upx=0,_mouth_upy=0,mouth_upx,mouth_upy;
        int _sellionx=0,_selliony=0,sellionx,selliony;
        int _eyebrow_rightx=0, _eyebrow_righty=0,eyebrow_rightx, eyebrow_righty;
        int _eyebrow_leftx=0, _eyebrow_lefty=0,eyebrow_leftx, eyebrow_lefty;
        int _mouth_center_upx=0, _mouth_center_upy=0,mouth_center_upx, mouth_center_upy;
        int _mouth_center_downx=0,_mouth_center_downy=0,mouth_center_downx,mouth_center_downy;
        int _mentonx=0, _mentony=0,mentonx, mentony;
        float _stomionx=0,_stomiony=0,stomionx,stomiony;
        double check1yl,check1yr;
        while(1)
        {
            // Grab a frame
            cv::Mat temp,gray,dst;
            cap >> dst;

            //flipping image (taking mirror image)
            temp = cv::Mat(dst.rows, dst.cols, CV_8UC3);
            cv::flip(dst, temp, 1);
            cvtColor( temp, gray, CV_BGR2GRAY );
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i)
            {
                full_object_detection shape = pose_model(cimg, faces[i]);
                cv::Rect l(shape.part(36).x()-5,shape.part(36).y()-20,(shape.part(39).x()-shape.part(36).x())+10,40);             
                cv::Rect r(shape.part(42).x()-5,shape.part(42).y()-20,(shape.part(45).x()-shape.part(42).x())+10,40);    
        
                nosex=shape.part(NOSE).x();nosey=shape.part(NOSE).y();
                right_eyex=shape.part(RIGHT_EYE).x(); right_eyey=shape.part(RIGHT_EYE).y();
                left_eyex=shape.part(LEFT_EYE).x(); left_eyey=shape.part(LEFT_EYE).y();
                right_sidex=shape.part(RIGHT_SIDE).x(); right_sidey=shape.part(RIGHT_SIDE).y();
                left_sidex=shape.part(LEFT_SIDE).x(); left_sidey=shape.part(LEFT_SIDE).y();
                mouth_rightx=shape.part(MOUTH_RIGHT).x(); mouth_righty=shape.part(MOUTH_RIGHT).y();
                mouth_leftx=shape.part(MOUTH_LEFT).x(); mouth_lefty=shape.part(MOUTH_LEFT).y();  
                mouth_downx=shape.part(MOUTH_DOWN).x(); mouth_downy=shape.part(MOUTH_DOWN).y();
                mouth_upx=shape.part(MOUTH_UP).x(); mouth_upy=shape.part(MOUTH_UP).y();
                sellionx=shape.part(SELLION).x(); selliony=shape.part(SELLION).y();
                eyebrow_rightx=shape.part(EYEBROW_RIGHT).x(); eyebrow_righty=shape.part(EYEBROW_RIGHT).y();
                eyebrow_leftx=shape.part(EYEBROW_LEFT).x(); eyebrow_lefty=shape.part(EYEBROW_LEFT).y();
                mouth_center_upx=shape.part(MOUTH_CENTER_TOP).x(); mouth_center_upy=shape.part(MOUTH_CENTER_TOP).y();
                mouth_center_downx=shape.part(MOUTH_CENTER_BOTTOM).x(); mouth_center_downy=shape.part(MOUTH_CENTER_BOTTOM).y();
                mentonx=shape.part(MENTON).x(); mentony=shape.part(MENTON).y();
                stomionx=(mouth_center_upx+mouth_center_downx)*0.5;
                stomiony=(mouth_center_upx+mouth_center_downx)*0.5;

                check1yl=(shape.part(39).x() + shape.part(36).x())*05;
                check1yr=(shape.part(45).x() + shape.part(42).x())*0.5;


                double dl=(shape.part(39).x()-shape.part(36).x());
                double dr=(shape.part(45).x()-shape.part(42).x());
        
                std::vector<cv::Point2f> detected_points;
                detected_points.push_back(cv::Point(sellionx,selliony));
                detected_points.push_back(cv::Point(right_eyex,right_eyey));
                detected_points.push_back(cv::Point(left_eyex,left_eyey));
                detected_points.push_back(cv::Point(right_sidex,right_sidey));
                detected_points.push_back(cv::Point(left_sidex,left_sidey));
                detected_points.push_back(cv::Point(mentonx,mentony));
                detected_points.push_back(cv::Point(nosex,nosey));
                detected_points.push_back(cv::Point(stomionx,stomiony));


    
                cv::solvePnP(cv::Mat(head_points),cv::Mat(detected_points),intrinsics, distortion, rvec, tvec, false,cv::ITERATIVE);
                cv::projectPoints(framePoints, rvec, tvec, intrinsics, distortion, imageFramePoints );
        
                double theta = cv::norm(rvec);
                double rx=rvec.at<double>(0,0);
                double ry=rvec.at<double>(1,0);
                double rz=rvec.at<double>(2,0);
        
                line(temp, cv::Point((int)imageFramePoints[0].x,(int)imageFramePoints[0].y), cv::Point((int)imageFramePoints[1].x,(int)imageFramePoints[1].y), cv::Scalar(255,0,0),2,8 );
                line(temp, cv::Point((int)imageFramePoints[0].x,(int)imageFramePoints[0].y), cv::Point((int)imageFramePoints[2].x,(int)imageFramePoints[2].y), cv::Scalar(0,255,0),2,8 );
                line(temp, cv::Point((int)imageFramePoints[0].x,(int)imageFramePoints[0].y), cv::Point((int)imageFramePoints[3].x,(int)imageFramePoints[3].y), cv::Scalar(0,0,255),2,8 );


                left=gray(l);right=gray(r);
                equalizeHist( left, left );     //histogram equalisation


                pupiltracker::findPupilEllipse(params, left, out1, log1); 
                pupiltracker::cvx::cross(left, out1.pPupil, 40, pupiltracker::cvx::rgb(255,255,0));
                cv::ellipse(left, out1.elPupil, pupiltracker::cvx::rgb(255,0,255));


       
                equalizeHist(right,right);   
                pupiltracker::findPupilEllipse(params, right, out, log); 
                pupiltracker::cvx::cross(right, out.pPupil, 40, pupiltracker::cvx::rgb(255,255,0));
                cv::ellipse(right, out.elPupil, pupiltracker::cvx::rgb(255,0,255));


                //Left Pupil center
                lX=out.elPupil.center.x+shape.part(42).x()-5;
                lY=out.elPupil.center.y+shape.part(42).y()-20;

                //Right Pupil Center
                rX=out1.elPupil.center.x+shape.part(36).x()-5;
                rY=out1.elPupil.center.y+shape.part(36).y()-20;


                cv::circle(temp,cv::Point(nosex,nosey),2,cv::Scalar(0,255,0),-1);//nose
                cv::circle(temp,cv::Point(right_eyex,right_eyey),2,cv::Scalar(0,255,0),-1);//right eye
                cv::circle(temp,cv::Point(left_eyex,left_eyey),2,cv::Scalar(0,255,0),-1);//left eye
                cv::circle(temp,cv::Point(right_sidex,right_sidey),2,cv::Scalar(0,255,0),-1);//right side
                cv::circle(temp,cv::Point(left_sidex,left_sidey),2,cv::Scalar(0,255,0),-1);//left side
                cv::circle(temp,cv::Point(eyebrow_rightx,eyebrow_righty),2,cv::Scalar(0,255,0),-1);//eyebrow right
                cv::circle(temp,cv::Point(eyebrow_leftx,eyebrow_lefty),2,cv::Scalar(155,255,200),-1);//eyebrow left
                cv::circle(temp,cv::Point(mouth_upx,mouth_upy),2,cv::Scalar(0,255,0),-1);//mouth up
                cv::circle(temp,cv::Point(mouth_downx,mouth_downy),2,cv::Scalar(0,255,0),-1);//mouth down
                cv::circle(temp,cv::Point(mouth_rightx,mouth_righty),2,cv::Scalar(0,255,0),-1);//mouth right
                cv::circle(temp,cv::Point(mouth_leftx,mouth_lefty),2,cv::Scalar(0,255,0),-1);//mouth left
                cv::circle(temp,cv::Point(sellionx,selliony),2,cv::Scalar(0,255,0),-1);//sellion
                cv::circle(temp,cv::Point(mouth_center_upx,mouth_center_upy),2,cv::Scalar(0,255,0),-1);//mouth center top
                cv::circle(temp,cv::Point(mouth_center_downx,mouth_center_downy),2,cv::Scalar(0,255,0),-1);//mouth center bottom
                cv::circle(temp,cv::Point(mentonx,mentony),2,cv::Scalar(0,255,0),-1);//Menton
        
                cv::circle(temp,cv::Point(lX,lY),5,cv::Scalar(0,0,255),0);
                cv::circle(temp,cv::Point(rX,rY),5,cv::Scalar(0,0,255),0);    

            }
            cv::imshow("image",temp);

            int k=cv::waitKey(1);
            if (k==27)
            {
              break;
            }
        }


        count++;
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

/*

pupiltracker::TrackerParams params;
        params.Radius_Min = 3;
        params.Radius_Max = 8;

        params.CannyBlur = 1;
        params.CannyThreshold1 = 20;
        params.CannyThreshold2 = 40;
        params.StarburstPoints = 0;

        params.PercentageInliers = 30;
        params.InlierIterations = 2;
        params.ImageAwareSupport = true;
        params.EarlyTerminationPercentage = 95;
        params.EarlyRejection = true;
        params.Seed = -1;

        */