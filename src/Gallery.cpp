#include "Gallery.hpp"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

Gallery::Gallery(FaceDetector& detector, FaceEncoder& encoder)
        : detector_(detector), encoder_(encoder) {}

bool Gallery::isImage(const std::string& ext)
{
    std::string e = ext;
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    return e == ".jpg" || e == ".jpeg" || e == ".png" || e == ".bmp";
}

size_t Gallery::load(const std::string& folder)
{
    if (!fs::is_directory(folder))
        throw std::runtime_error("Папка не найдена: " + folder);

    entries_.clear();
    skipped_.clear();

    for (const auto& item : fs::directory_iterator(folder))
    {
        if (!item.is_regular_file() || !isImage(item.path().extension().string()))
            continue;

        const std::string path = item.path().string();
        cv::Mat img = cv::imread(path);
        if (img.empty())
        {
            skipped_.push_back(path);
            continue;
        }
        
        cv::Mat face;
        if (!detector_.detectBest(img, face))
        {
            skipped_.push_back(path);
            continue;
        }

        entries_.push_back({path, encoder_.encode(img,  face)});
    }
    return entries_.size();
}