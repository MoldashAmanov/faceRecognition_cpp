#include "FaceDetector.hpp"
#include <stdexcept>

FaceDetector::FaceDetector(const std::string& modelPath, float scoreThreshold,
                            float nmsTreshold, int topK)
{
    detector_ = cv::FaceDetectorYN::create(
        modelPath, "", cv::Size(320, 320), scoreThreshold, nmsTreshold, topK);
    if (detector_.empty())
        throw std::runtime_error("Не удалось загрузить модель детектора: " + modelPath);
}

cv::Mat FaceDetector::detect(const cv::Mat& image)
{
    detector_->setInputSize(image.size()); // Размер входа = размер кадра
    cv::Mat faces;
    detector_->detect(image, faces);
    return faces;
}

bool FaceDetector::detectBest(const cv::Mat& image, cv::Mat& faceRow)
{
    cv::Mat faces = detect(image);
    if (faces.empty() || faces.rows == 0)
        return false;

    int best = 0;
    float bestScore = faces.at<float>(0, 14);
    for (int i = 1; i < faces.rows; ++i)
    {
        float s = faces.at<float>(i, 14);
        if (s > bestScore)
        {
            bestScore = s;
            best = i;
        }
    }
    faceRow = faces.row(best).clone();
    return true;
}