#include "utils.hpp"
#include "dataStruct.hpp"
#include "cameraCalibration.hpp"
#include "poseEstimation.hpp"
#include "imageUndistortion.hpp"
#include <opencv2/core/base.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "opencv2/core/utility.hpp"

#include <stdio.h>
static void saveXYZ(const char* filename, const cv::Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            cv::Vec3f point = mat.at<cv::Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

int main(int argc, char** argv) {
    const char* param1_opt = "--param1=";
    const char* param2_opt = "--param2=";
    const char* settings1_opt = "--settings1=";
    const char* settings2_opt = "--settings2=";
    const char* im1_opt = "--im1=";
    const char* im2_opt = "--im2=";

    int minDisparity = 0;
    int numDisparities = 64;
    int blockSize = 25;
    int disp12MaxDiff = 10;
    int uniquenessRatio = 10;
    int speckleWindowSize = 50;
    int speckleRange = 4;

    cv::String settingsFilename1 = "data/calib1.xml";
    cv::String parametersFilename1 = "data/calib1param.xml";
    cv::String settingsFilename2 = "data/calib2.xml";
    cv::String parametersFilename2 = "data/calib2param.xml";
    cv::String imgLstr = "data/object/cam1/3.jpg";
    cv::String imgRstr = "data/object/cam2/3.jpg";


    Settings settings1,settings2;
    if (!openData(settingsFilename1, settings1)) {
        return -1;
    }
    if (!openData(settingsFilename2, settings2)) {
        return -1;
    }

    auto images1 = std::vector<cv::Mat>{};
    openImages(settings1.inputsPattern, images1);
    auto images2 = std::vector<cv::Mat>{};
    openImages(settings2.inputsPattern, images2);

    CameraParameters parameters1,parameters2;
    CornersSet cornersSet1, cornersSet2;

    if (!camerasCalibration(images1, settings1, parameters1, cornersSet1, images2, settings2, parameters2, cornersSet2, false, false)) {
        return -1;
    }
    if (!saveData(parametersFilename1, parameters1)) {
        return -1;
    }
    if (!saveData(parametersFilename2, parameters2)) {
        return -1;
    }
    cv::Mat Rmat, Tmat, Emat, Fmat;
	cv::Mat newL, newR;

    newL = cv::getOptimalNewCameraMatrix(parameters1.cameraMatrix, parameters1.distCoeffs, 
        cornersSet1.imageSize, 1, cornersSet1.imageSize,0);
    newR = cv::getOptimalNewCameraMatrix(parameters2.cameraMatrix, parameters2.distCoeffs, 
        cornersSet2.imageSize, 1, cornersSet2.imageSize,0);

    int flag = 0;
	flag |= cv::CALIB_FIX_INTRINSIC;
    flag |= cv::CALIB_FIX_ASPECT_RATIO;
    cv::stereoCalibrate(cornersSet1.objectsPoints, cornersSet1.imagesPoints, cornersSet2.imagesPoints,
        newL, parameters1.distCoeffs, newR, parameters2.distCoeffs, cornersSet1.imageSize,
        Rmat, Tmat, Emat, Fmat, flag, cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 1e-10));

    cv::Mat rect_l, rect_r, proj_mat_l, proj_mat_r, Q;
    cv::stereoRectify(newL, parameters1.distCoeffs, newR, parameters2.distCoeffs,
        cornersSet1.imageSize, Rmat, Tmat, rect_l, rect_r, proj_mat_l, proj_mat_r, Q, 1);

    cv::Mat Left_Stereo_Map1, Left_Stereo_Map2;
    cv::Mat Right_Stereo_Map1, Right_Stereo_Map2;

    cv::initUndistortRectifyMap(newL, parameters1.distCoeffs, rect_l, proj_mat_l, cornersSet1.imageSize, CV_16SC2, Left_Stereo_Map1, Left_Stereo_Map2);
    cv::initUndistortRectifyMap(newR, parameters2.distCoeffs, rect_r, proj_mat_r, cornersSet2.imageSize, CV_16SC2, Right_Stereo_Map1, Right_Stereo_Map2);


    cv::Mat imgLeft = cv::imread(imgLstr, cv::IMREAD_GRAYSCALE);
    cv::Mat imgRight = cv::imread(imgRstr, cv::IMREAD_GRAYSCALE);
    cv::Mat remapedL, remapedR;

    cv::remap(imgLeft, remapedL, Left_Stereo_Map1, Left_Stereo_Map2, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
    cv::remap(imgRight, remapedR, Right_Stereo_Map1, Right_Stereo_Map2, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
    //cv::resize(remapedL, remapedL, cv::Size(), 0.5, 0.5);
    //cv::resize(remapedR, remapedR, cv::Size(), 0.5, 0.5);
    

    cv::waitKey(0);

    //cv::Mat imgLeft = cv::imread("data/batleft.jpg", cv::IMREAD_GRAYSCALE);
    //cv::Mat imgRight = cv::imread("data/batright.jpg", cv::IMREAD_GRAYSCALE); 

    cv::resize(remapedL, remapedL,cv::Size(),0.3,0.3);
    cv::resize(remapedR, remapedR,cv::Size(),0.3,0.3);
    cv::imshow("remapedL", remapedL);
    cv::imshow("remapedR", remapedR);

    std::vector<cv::KeyPoint> kpL, kpR;
    cv::Mat descL, descR; 
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->detectAndCompute(remapedL, cv::noArray(),kpL, descL);
    sift->detectAndCompute(remapedR, cv::noArray(),kpR, descR);

    cv::Mat imgLSift, imgRSift;
    cv::drawKeypoints(remapedL, kpL, imgLSift);
    cv::drawKeypoints(remapedR, kpR, imgRSift);
    imshow("Left Sift", imgLSift);
    imshow("Right Sift", imgRSift);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch(descL, descR, knn_matches, 2);

    const float ratio_thresh = 0.55f;
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> pts1, pts2;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
            pts2.push_back(kpR[knn_matches[i][0].trainIdx].pt);
            pts1.push_back(kpL[knn_matches[i][0].queryIdx].pt);
        }
    }
    
    cv::Mat img_matches;
    drawMatches( remapedL, kpL, remapedR, kpR, good_matches, img_matches, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imshow("Good Matches", img_matches);

    
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(minDisparity,numDisparities,blockSize,
    disp12MaxDiff,uniquenessRatio,speckleWindowSize,speckleRange);
    cv::Mat out, resi;
    stereo->compute(remapedL, remapedR, out);
    cv::normalize(out, out, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::Mat out8;
    out.convertTo(out8, CV_8U, 255/(16*16.));
    imshow("out8", out8);
    cv::Mat xyz;
    reprojectImageTo3D(out8, xyz, Q, true);
    saveXYZ("data/cloud.txt", xyz);
    cv::waitKey(0);
    return 0;
}