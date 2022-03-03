#pragma once

#include <iostream>
#include <vector>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

#include "types.hpp"

struct Settings {
    bool isGood;
    cv::Size boardSize; // The size of the board -> Number of items by width and height
    float squareSize; // The size of a square in your defined unit (point, millimeter,etc).
    cv::String inputsPattern; // The inputs

    Settings() : isGood(false), boardSize(), squareSize(-1.f), inputsPattern() {}

    void read(const cv::FileNode &node) {
        node["BoardSize_Width"] >> boardSize.width;
        node["BoardSize_Height"] >> boardSize.height;
        node["Square_Size"] >> squareSize;
        node["InputsPattern"] >> inputsPattern;
        interpret();
    }

    void interpret() {
        if (boardSize.width <= 0 || boardSize.height <= 0) {
            std::cerr << "Invalid Board size: " << boardSize.width << " " << boardSize.height << std::endl;
            return;
        }
        if (squareSize <= 10e-6) {
            std::cerr << "Invalid square size " << squareSize << std::endl;
            return;
        }
        if (inputsPattern.empty()) {
            std::cerr << " Non-existent input: " << inputsPattern << std::endl;
            return;
        }

        isGood = true;
    }
};

static void read(const cv::FileNode &node_, Settings &settings_, const Settings &defaultValue_ = Settings()) {
    if (node_.empty())
        settings_ = defaultValue_;
    else
        settings_.read(node_);
}

static bool openData(const cv::String &storageFilename_, Settings &data_) {
    auto storage = cv::FileStorage{storageFilename_, cv::FileStorage::READ};
    if (!storage.isOpened()) {
        std::cerr << "Could not open the file: \"" << storageFilename_ << "\"" << std::endl;
        return false;
    }

    storage["Settings"] >> data_;
    storage.release();
    return true;
}

struct CornersSet {
    cv::Size imageSize;
    std::vector<std::vector<cv::Point3f>> objectsPoints; // 3d point in real world space
    std::vector<std::vector<cv::Point2f>> imagesPoints; // 2d points in image plane.

    CornersSet() = default;

    void read(const cv::FileNode &node) {
        node["ImageSize"] >> imageSize;
        node["ObjectsPoints"] >> objectsPoints;
        node["ImagesPoints"] >> imagesPoints;
    }

    void write(cv::FileStorage &storage) const {
        storage << "{" << "ImageSize" << imageSize
                << "ObjectsPoints" << objectsPoints
                << "ImagesPoints" << imagesPoints << "}";
    }
};

static void read(const cv::FileNode &node_, CornersSet &cornersSet_,
                 const CornersSet &defaultValue_ = CornersSet()) {
    if (node_.empty())
        cornersSet_ = defaultValue_;
    else
        cornersSet_.read(node_);
}

static void write(cv::FileStorage &storage_, std::string &, const CornersSet &cornersSet_) {
    cornersSet_.write(storage_);
}

static bool openData(const cv::String &storageFilename_, CornersSet &data_) {
    auto storage = cv::FileStorage{storageFilename_, cv::FileStorage::READ};
    if (!storage.isOpened()) {
        std::cerr << "Could not open the file: \"" << storageFilename_ << "\"" << std::endl;
        return false;
    }

    storage["CornersSet"] >> data_;
    storage.release();
    return true;
}

static bool saveData(const cv::String &storageFilename_, const CornersSet &data_) {
    auto storage = cv::FileStorage{storageFilename_, cv::FileStorage::WRITE};
    if (!storage.isOpened()) {
        std::cerr << "Could not open the file: \"" << storageFilename_ << "\"" << std::endl;
        return false;
    }

    storage << "CornersSet" << data_;
    storage.release();
    return true;
}

struct CameraParameters {
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    std::vector<double> reprojErrors;
    double totalAvgError;

    CameraParameters() : rvecs(), tvecs(), reprojErrors(), totalAvgError(-1.) {
        cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
    }

    void read(const cv::FileNode &node) {
        node["CameraMatrix"] >> cameraMatrix;
        node["DistCoeffs"] >> distCoeffs;
        node["RotationVectors"] >> rvecs;
        node["TranslationVectors"] >> tvecs;
    }

    void write(cv::FileStorage &storage) const {
        storage << "{" << "CameraMatrix" << cameraMatrix
                << "DistCoeffs" << distCoeffs
                << "RotationVectors" << rvecs
                << "TranslationVectors" << tvecs << "}";
    }
};

static void read(const cv::FileNode &node_, CameraParameters &parameters_,
                 const CameraParameters &defaultValue_ = CameraParameters()) {
    if (node_.empty())
        parameters_ = defaultValue_;
    else
        parameters_.read(node_);
}

static void write(cv::FileStorage &storage_, std::string &, const CameraParameters &parameters_) {
    parameters_.write(storage_);
}

static bool openData(const cv::String &storageFilename_, CameraParameters &data_) {
    auto storage = cv::FileStorage{storageFilename_, cv::FileStorage::READ};
    if (!storage.isOpened()) {
        std::cerr << "Could not open the file: \"" << storageFilename_ << "\"" << std::endl;
        return false;
    }

    storage["CameraParameters"] >> data_;
    storage.release();
    return true;
}

static bool saveData(const cv::String &storageFilename_, const CameraParameters &data_) {
    auto storage = cv::FileStorage{storageFilename_, cv::FileStorage::WRITE};
    if (!storage.isOpened()) {
        std::cerr << "Could not open the file: \"" << storageFilename_ << "\"" << std::endl;
        return false;
    }

    storage << "CameraParameters" << data_;
    storage.release();
    return true;
}