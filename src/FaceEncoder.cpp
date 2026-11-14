#include "FaceEncoder.hpp"
#include <stdexcept>

FaceEncoder::FaceEncoder(const std::string& modelPath)
{
    recognizer_ = cv::FaceRecognizerSF::create(modelPath, "");
    if (recognizer_.empty())
    {
        throw std::runtime_error("Не удалось загрузить модель распознования: " + modelPath);
    }
}

cv::Mat FaceEncoder::encode(const cv::Mat& image, const cv::Mat& faceRow)
{
    cv::Mat aligned, feature;
    recognizer_->alignCrop(image, faceRow, aligned);
    recognizer_->feature(aligned, feature);
    return feature.clone();
}

double FaceEncoder::similarity(const cv::Mat& a, const cv::Mat& b) const
{
    return recognizer_->match(a, b, cv::FaceRecognizerSF::DisType::FR_COSINE);
}