#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include "types.hpp"
#include "dataStruct.hpp"

static void drawPointAxis(cv::Mat &image_, const std::vector<cv::Point2f> &corners,
                          const std::vector<cv::Point2f> &imagePoints) {
    const auto &corner = corners[0];
    cv::line(image_, corner, imagePoints[0], cv::Scalar{255., 0., 0.}, 5);
    cv::line(image_, corner, imagePoints[1], cv::Scalar{0., 255., 0.}, 5);
    cv::line(image_, corner, imagePoints[2], cv::Scalar{0., 0., 255.}, 5);
}

static bool poseEstimation(const cv::Mat& image_, const Settings &settings_, const CameraParameters &parameters_,
                           const std::vector<cv::Point3f>& objP_, cv::TermCriteria criteria_,
                           std::vector<cv::Point2f>& corners_, cv::Mat &rvec_, cv::Mat &tvec_) {
    bool found = cv::findChessboardCorners(image_, settings_.boardSize, corners_,
                                           cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK |
                                           cv::CALIB_CB_NORMALIZE_IMAGE);

    if (found) {
        cv::cornerSubPix(image_, corners_, cv::Size{11, 11},
                         cv::Size{-1, -1}, criteria_);

        cv::solvePnPRansac(objP_, corners_, parameters_.cameraMatrix,
                           parameters_.distCoeffs, rvec_, tvec_);
    }

    return found;
}

static bool poseEstimation(const cv::Mat& image_, const Settings &settings_, const CameraParameters &parameters_,
                           cv::Mat &rvec_, cv::Mat &tvec_) {
    auto criteria = cv::TermCriteria{cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                     30, 0.1};
    auto objP = std::vector<cv::Point3f>{};
    for (int i = 0; i < settings_.boardSize.height; i++) {
        for (int j = 0; j < settings_.boardSize.width; j++) {
            objP.emplace_back(float(j) * settings_.squareSize, float(i) * settings_.squareSize, 0);
        }
    }
    std::vector<cv::Point2f> corners;

    cv::Mat gray;
    cv::cvtColor(image_, gray, cv::COLOR_BGR2GRAY);

    return poseEstimation(gray, settings_, parameters_, objP, criteria, corners, rvec_, tvec_);
}

static void poseEstimation(const std::vector<cv::Mat> &images_, const Settings &settings_,
                           CameraParameters &parameters_,
                           bool display_ = false) {
    assert(!images_.empty());
    assert(settings_.isGood);

    auto criteria = cv::TermCriteria{cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                     30, 0.1};
    auto objP = std::vector<cv::Point3f>{};
    for (int i = 0; i < settings_.boardSize.height; i++) {
        for (int j = 0; j < settings_.boardSize.width; j++) {
            objP.emplace_back(float(j) * settings_.squareSize, float(i) * settings_.squareSize, 0);
        }
    }
    float axisLength = 3.f * settings_.squareSize;
    auto axis = std::vector<cv::Point3f>{
            cv::Point3f{axisLength, 0, 0},
            cv::Point3f{0, axisLength, 0},
            cv::Point3f{0, 0, -axisLength}
    };
    parameters_.rvecs.clear();
    parameters_.tvecs.clear();

    cv::Size imageSize = images_[0].size();
    for (auto image: images_) {
        assert(image.size() == imageSize);
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        cv::Mat rvec, tvec;
        bool found = poseEstimation(gray, settings_, parameters_, objP, criteria, corners, rvec, tvec);
        if(found) {
            parameters_.rvecs.push_back(rvec);
            parameters_.tvecs.push_back(tvec);

            if (display_) {
                cv::Mat imagePoints2;
                projectPoints(axis, rvec, tvec,
                              parameters_.cameraMatrix,
                              parameters_.distCoeffs, imagePoints2);
                drawPointAxis(image, corners, imagePoints2);
                cv::imshow("Image", image);
                cv::waitKey(500);
            }
        }
    }
    if (display_) {
        cv::destroyAllWindows();
    }
}