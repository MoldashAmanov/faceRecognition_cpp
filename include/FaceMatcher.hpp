#pragma once
#include "FaceEncoder.hpp"
#include "Gallery.hpp"
#include <string>
#include <vector>

struct MatchResult
{
    std::string path;
    double score;
    bool isMatch;
};

class FaceMatcher
{
    public:
        FaceMatcher(FaceEncoder& encoder, const Gallery& gallery, double threshold = 0.363);

        std::vector<MatchResult> match(const cv::Mat& queryFeature) const;
        bool best(const cv::Mat& queryFeature, MatchResult& out) const;

        double threshold() const
        {
            return threshold_;
        }
    private:
        FaceEncoder& encoder_;
        const Gallery& gallery_;
        double threshold_;
};
