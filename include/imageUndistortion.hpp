#pragma once

#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include "types.hpp"
#include "dataStruct.hpp"

static void undistort(const cv::Mat &image_, const CameraParameters &parameter_, cv::Mat &undistortedImage_) {
    cv::Size imgSize = image_.size();
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(
            parameter_.cameraMatrix, parameter_.distCoeffs, imgSize, 1, imgSize
    );

    cv::undistort(image_, undistortedImage_, parameter_.cameraMatrix,
                  parameter_.distCoeffs, newCameraMatrix);
}

static void undistort(const std::vector<cv::Mat> &images_, const CameraParameters &parameter_,
                      std::vector<cv::Mat> &undistortedImages_) {
    uint n = images_.size();
    undistortedImages_.resize(n);

    for (uint i = 0u; i < n; i++) {
        const auto &image = images_[i];
        auto &undistortedImage = undistortedImages_[i];
        undistort(image, parameter_, undistortedImage);
    }
}