#pragma once
#include "FaceDetector.hpp"
#include "FaceEncoder.hpp"
#include "FaceMatcher.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

// Результат анализа одного лица, который можно рисовать повторно
struct FaceView
{
    cv::Rect box;
    std::string label;
    bool match;
};

class VideoMatcher
{
    public:
        // recomputeEvery - как часто пересчитывать (1 - каждый кадр)
        // confirmFrames - сколько подряд пересчетов с совпадением нужно для баннера
        VideoMatcher(FaceDetector& detector, FaceEncoder& encoder, const FaceMatcher& matcher,
                    int recomputeEvery = 3, int confirmFrames = 3);

        // source: индекс камеры ("0") или путь к видеофайлу
        void run(const std::string& source);
    private:
        std::vector<FaceView> analyze(const cv::Mat& frame);
        void draw(cv::Mat& frame) const;
        static std::string nameFromPath(const std::string& path);

        FaceDetector& detector_;
        FaceEncoder& encoder_;
        const FaceMatcher& matcher_;
        int recomputeEvery_;
        int confirmFrames_;

        std::vector<FaceView> views_; // кэш последнего анализа
        int streak_ = 0;              // подряд пересчетов с совпадением
};