#pragma once

#include <vector>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include "types.hpp"
#include "dataStruct.hpp"

static void setup(const std::vector<cv::Mat> &images_, const Settings &settings_,
                  CornersSet &cornersSet_, bool display_ = false) {
    assert(!images_.empty());
    assert(settings_.isGood);

    auto criteria = cv::TermCriteria{cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1};
    auto objP = std::vector<cv::Point3f>{};
    for (int i = 0; i < settings_.boardSize.height; i++) {
        for (int j = 0; j < settings_.boardSize.width; j++) {
            objP.emplace_back(float(j) * settings_.squareSize, float(i) * settings_.squareSize, 0);
        }
    }

    cornersSet_.imageSize = images_[0].size();
    for (auto image: images_) {
        assert(image.size() == cornersSet_.imageSize);
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, settings_.boardSize, corners,
                                               cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK |
                                               cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size{11, 11},
                             cv::Size{-1, -1}, criteria);
            cornersSet_.imagesPoints.push_back(corners);

            if (display_) {
                cv::drawChessboardCorners(image, settings_.boardSize, corners, found);
                cv::imshow("Image", image);
                cv::waitKey(500);
            }
        }
    }
    if (display_) {
        cv::destroyAllWindows();
    }

    cornersSet_.objectsPoints.resize(cornersSet_.imagesPoints.size(), objP);
}
static void stereoSetup(const std::vector<cv::Mat> &images1_, const Settings &settings1_,
                  CornersSet &cornersSet1_, const std::vector<cv::Mat> &images2_, const Settings &settings2_,
                  CornersSet &cornersSet2_, bool display_ = false) {
    assert(!images1_.empty());
    assert(!images2_.empty());
    assert(images1_.size() == images2_.size());
    assert(settings1_.isGood);

    auto criteria = cv::TermCriteria{cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001};
    auto objP = std::vector<cv::Point3f>{};
    for (int i = 0; i < settings1_.boardSize.height; i++) {
        for (int j = 0; j < settings1_.boardSize.width; j++) {
            objP.emplace_back(float(j) * settings1_.squareSize, float(i) * settings1_.squareSize, 0);
        }
    }
    
    cornersSet1_.imageSize = images1_[0].size();
    cornersSet2_.imageSize = images2_[0].size();

    for (int i = 0; i < images1_.size(); ++i) {
        cv::Mat grayL, grayR;
        std::vector<cv::Point2f> cornersL, cornersR;
        bool foundL, foundR;
        cv::cvtColor(images1_[i], grayL, cv::COLOR_BGR2GRAY);
        cv::cvtColor(images2_[i], grayR, cv::COLOR_BGR2GRAY);

        foundL = cv::findChessboardCorners(grayL, settings1_.boardSize, cornersL,
                                               cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK |
                                               cv::CALIB_CB_NORMALIZE_IMAGE);
        foundR = cv::findChessboardCorners(grayR, settings2_.boardSize, cornersR,
                                               cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK |
                                               cv::CALIB_CB_NORMALIZE_IMAGE);
        if (foundL && foundR) {
            cv::cornerSubPix(grayL, cornersL, cv::Size{11, 11},
                             cv::Size{-1, -1}, criteria);
            cv::cornerSubPix(grayR, cornersR, cv::Size{11, 11},
                             cv::Size{-1, -1}, criteria);
            cornersSet1_.imagesPoints.push_back(cornersL);
            cornersSet2_.imagesPoints.push_back(cornersR);
            if (display_) {
                cv::Mat resL, resR;
                cv::drawChessboardCorners(images1_[i], settings1_.boardSize, cornersL, foundL);
                cv::drawChessboardCorners(images2_[i], settings2_.boardSize, cornersR, foundR);
                cv::resize(images1_[i], resL, cv::Size(), 0.5, 0.5);
                cv::resize(images2_[i], resR, cv::Size(), 0.5, 0.5);
                cv::imshow("Image 1", resL);
                cv::imshow("Image 2", resR);
                cv::waitKey(500);
            }
        }
    }
    if (display_) {
        cv::destroyAllWindows();
    }
    cornersSet1_.objectsPoints.resize(cornersSet1_.imagesPoints.size(), objP);
    cornersSet2_.objectsPoints.resize(cornersSet2_.imagesPoints.size(), objP);
}
static void computeReprojectionErrors(const CornersSet &cornersSet_, CameraParameters &parameters_,
                                      bool verbose_ = false) {
    cv::Mat imagePoints2;
    uint totalPoints = 0u;
    double totalErr = 0;
    parameters_.reprojErrors.resize(cornersSet_.imagesPoints.size());

    for (uint i = 0u; i < cornersSet_.objectsPoints.size(); ++i) {
        auto objectPoints = cv::Mat{cornersSet_.objectsPoints[i]};
        auto imagePoints = cv::Mat{cornersSet_.imagesPoints[i]};
        projectPoints(objectPoints, parameters_.rvecs[i], parameters_.tvecs[i],
                      parameters_.cameraMatrix,
                      parameters_.distCoeffs, imagePoints2);
        double err = cv::norm(imagePoints, imagePoints2, cv::NORM_L2);
        uint n = cornersSet_.objectsPoints[i].size();
        parameters_.reprojErrors[i] = std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;

        if (verbose_) {
            std::cout << "i: [" << i << "]" << std::endl;
            std::cout << "\tError: " << err << std::endl;
            std::cout << "\tReprojection error: " << parameters_.reprojErrors[i] << std::endl;
            std::cout << "\tTotal error: " << totalErr << std::endl;
            std::cout << "\tTotal Points: " << totalPoints << std::endl;
        }
    }

    parameters_.totalAvgError = std::sqrt(totalErr / totalPoints);
    if (verbose_) {
        std::cout << "Total average error: " << parameters_.totalAvgError << std::endl;
    }
}

static bool calibrate(const CornersSet &cornersSet_,
                      CameraParameters &parameters_, bool verbose_ = false) {
    double rms = cv::calibrateCamera(cornersSet_.objectsPoints, cornersSet_.imagesPoints,
                                     cornersSet_.imageSize,
                                     parameters_.cameraMatrix, parameters_.distCoeffs,
                                     parameters_.rvecs, parameters_.tvecs);

    bool calibrationSuccess = cv::checkRange(parameters_.cameraMatrix) && cv::checkRange(parameters_.distCoeffs);
    if (verbose_) {
        std::cout << "Re-projection error reported by calibrateCamera: " << rms << std::endl;
        std::cout << (calibrationSuccess ? "Calibration succeeded." : "Calibration failed.") << std::endl <<
            "Total average reprojection error = " << parameters_.totalAvgError << std::endl;
    }

    return calibrationSuccess;
}

static bool cameraCalibration(const std::vector<cv::Mat> &images_, const Settings &settings_,
                              CameraParameters &parameters_, CornersSet& cornersSet, bool verbose_ = false, bool display_ = false) {
    setup(images_, settings_, cornersSet, display_);

    bool calibrationSuccess = calibrate(cornersSet, parameters_, verbose_);
    if (!calibrationSuccess) {
        return false;
    }

    computeReprojectionErrors(cornersSet, parameters_, verbose_);

    return true;
}
static bool camerasCalibration(const std::vector<cv::Mat> &images1_, const Settings &settings1_,
                              CameraParameters &parameters1_, CornersSet& cornersSet1_,const std::vector<cv::Mat> &images2_, const Settings &settings2_,
                              CameraParameters &parameters2_, CornersSet& cornersSet2_, bool verbose_ = false, bool display_ = false) {
    std::cout << "stereo setup" << std::endl;
    stereoSetup(images1_, settings1_, cornersSet1_, images2_, settings2_, cornersSet2_, display_);

    std::cout << "calib L" << std::endl;
    bool calibrationSuccessL = calibrate(cornersSet1_, parameters1_, verbose_);
    std::cout << "calib R" << std::endl;
    bool calibrationSuccessR = calibrate(cornersSet2_, parameters2_, verbose_);
    if (!calibrationSuccessL || !calibrationSuccessR) {
        return false;
    }
    std::cout << "reproj L" << std::endl;
    computeReprojectionErrors(cornersSet1_, parameters1_, verbose_);
    std::cout << "reproj R" << std::endl;
    computeReprojectionErrors(cornersSet2_, parameters2_, verbose_);

    return true;
}