#include "FaceMatcher.hpp"
#include <algorithm>

FaceMatcher::FaceMatcher(FaceEncoder& encoder, const Gallery& gallery, double threshold)
            : encoder_(encoder), gallery_(gallery), threshold_(threshold) {}

std::vector<MatchResult> FaceMatcher::match(const cv::Mat& queryFeature) const
{
    std::vector<MatchResult> results;
    results.reserve(gallery_.entries().size());

    for (const auto& e : gallery_.entries())
    {
        double s = encoder_.similarity(queryFeature, e.feature);
        results.push_back({e.path, s, s >= threshold_});
    }
    std::sort(results.begin(), results.end(), 
            [](const MatchResult& a, const MatchResult& b)
        {
            return a.score > b.score;
        } 
    );
    return results;
}

bool FaceMatcher::best(const cv::Mat& queryFeature, MatchResult& out) const
{
    if (gallery_.entries().empty())
        return false;
    
    double bestScore = -2.0;    // косинус лежит в [-1; 1]
    const GalleryEntry* bestEntry = nullptr;
    for (const auto& e : gallery_.entries())
    {
        double s = encoder_.similarity(queryFeature, e.feature);
        if (s > bestScore)
        {
            bestScore = s;
            bestEntry = &e;
        }
    }
    out = {bestEntry->path, bestScore, bestScore >= threshold_};
    return true;
}