#pragma once
#include <opencv2/objdetect/face.hpp>
#include <opencv2/core.hpp>
#include <string>

class FaceEncoder
{
    public:
        explicit FaceEncoder(const std::string& modelPath);

        cv::Mat encode(const cv::Mat& image, const cv::Mat& faceRow);
        double similarity(const cv::Mat& a, const cv::Mat& b) const;
    private:
        cv::Ptr<cv::FaceRecognizerSF> recognizer_;
};