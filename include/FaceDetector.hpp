#pragma once
#include <opencv2/objdetect/face.hpp>
#include <opencv2/core.hpp>
#include <string>

class FaceDetector
{
    public:
        explicit FaceDetector(const std::string& modelPath,
                                    float scoreThreshold = 0.9f,
                                    float mnsThreshold = 0.3f,
                                    int topK = 5000);
        
        // Все лица: матрица Nx15 (пустая, если лиц нет)
        cv::Mat detect(const cv::Mat& image);

        // Самое уверенное лицо; false, если нет лиц
        bool detectBest(const cv::Mat& image, cv::Mat& faceRow);
    private:
        cv::Ptr<cv::FaceDetectorYN> detector_;
};