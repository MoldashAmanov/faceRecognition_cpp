#pragma once
#include "FaceEncoder.hpp"
#include "FaceDetector.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct GalleryEntry
{
    std::string path;
    cv::Mat feature;
};

class Gallery
{
    public:
        Gallery(FaceDetector& detector, FaceEncoder& encoder);

        size_t load(const std::string& folder);

        const std::vector<GalleryEntry>& entries() const
        {
            return entries_;
        }
        const std::vector<std::string>& skipped() const
        {
            return skipped_;
        }
    private:
        static bool isImage(const std::string& ext);

        FaceDetector& detector_;
        FaceEncoder& encoder_;
        std::vector<GalleryEntry> entries_;
        std::vector<std::string> skipped_;
};