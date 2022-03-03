#pragma once

#include <vector>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>

#include "dataStruct.hpp"

static void openImages(const cv::String &inputsPattern_, std::vector<cv::Mat> &images_) {
    auto filesNames = std::vector<cv::String>{};
    cv::glob(inputsPattern_, filesNames, false);

    uint nFile = filesNames.size();
    images_.resize(nFile);
    for (uint i = 0u; i < nFile; i++) {
        images_[i] = cv::imread(filesNames[i]);
    }
}

static void writeImages(const cv::String& foldername_, const std::vector<cv::Mat> &images_) {
    uint i=0u;
    for(const auto& image : images_) {
        cv::String filename = foldername_ + std::to_string(i) + ".png";
        cv::imwrite(filename, image);

        i++;
    }
}