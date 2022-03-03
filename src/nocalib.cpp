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

int main() {
    
    cv::Mat imgLeft = cv::imread("data/im2.png", cv::IMREAD_GRAYSCALE);
    cv::Mat imgRight = cv::imread("data/im6.png", cv::IMREAD_GRAYSCALE);

    cv::imshow("imgL", imgLeft);
    cv::imshow("imgR", imgRight);

    std::vector<cv::KeyPoint> kpL, kpR;
    cv::Mat descL, descR; 
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->detectAndCompute(imgLeft, cv::noArray(),kpL, descL);
    sift->detectAndCompute(imgRight, cv::noArray(),kpR, descR);

    cv::Mat imgLSift, imgRSift;
    cv::drawKeypoints(imgLeft, kpL, imgLSift);
    cv::drawKeypoints(imgRight, kpR, imgRSift);
    imshow("Left Sift", imgLSift);
    imshow("Right Sift", imgRSift);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch(descL, descR, knn_matches, 2);

    const float ratio_thresh = 0.45f;
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
    drawMatches( imgLeft, kpL, imgRight, kpR, good_matches, img_matches, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imshow("Good Matches", img_matches);

    int minDisparity = 0;
    int numDisparities = 64;
    int blockSize = 15;
    int disp12MaxDiff = 2;
    int uniquenessRatio = 5;
    int speckleWindowSize = 50;
    int speckleRange = 2;
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(minDisparity,numDisparities,blockSize,
    disp12MaxDiff,uniquenessRatio,speckleWindowSize,speckleRange);
    cv::Mat out, resi;
    stereo->compute(imgLeft, imgRight, out);
    cv::normalize(out, out, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    imshow("out", out);
    cv::Mat xyz;
    cv::Mat Q(4, 4, CV_32F);
    Q.at<float>(0, 0) = 1;
    Q.at<float>(1, 1) = -1;
    Q.at<float>(2, 2) = 0.5;
    Q.at<float>(3, 3) = 1;
    reprojectImageTo3D(out, xyz, Q, true);
    saveXYZ("data/cloud.txt", xyz);
    cv::waitKey(0);
    return 0;
}