#include "VideoMatcher.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <iostream>

VideoMatcher::VideoMatcher(FaceDetector& d, FaceEncoder& e, const FaceMatcher& m,
                            int recomputeEvery, int confirmFrames)
            : detector_(d), encoder_(e), matcher_(m),
            recomputeEvery_(std::max(1, recomputeEvery)),
            confirmFrames_(std::max(1, confirmFrames)) {}

std::string VideoMatcher::nameFromPath(const std::string& path)
{
    return std::filesystem::path(path).stem().string();
}

std::vector<FaceView> VideoMatcher::analyze(const cv::Mat& frame)
{
    std::vector<FaceView> result;
    if (frame.empty()) return result;

    const int targetWidth = 640;
    double scale = static_cast<double>(targetWidth) / frame.cols;
    cv::Mat small;
    cv::resize(frame, small, cv::Size(), scale, scale);

    cv::Mat faces = detector_.detect(small);

    if (faces.empty() || faces.rows == 0)
        return result;

    for (int i = 0; i < faces.rows; ++i)
    {
        cv::Mat smallRow = faces.row(i);

        float score = smallRow.at<float>(0, 14);
        if (score < 0.5f) continue;

        float x = smallRow.at<float>(0, 0) / scale;
        float y = smallRow.at<float>(0, 1) / scale;
        float w = smallRow.at<float>(0, 2) / scale;
        float h = smallRow.at<float>(0, 3) / scale;

        cv::Rect box(
            static_cast<int>(x), static_cast<int>(y),
            static_cast<int>(w), static_cast<int>(h)
        );

        cv::Mat bigRow = smallRow.clone();
        bigRow.at<float>(0, 0) = x;
        bigRow.at<float>(0, 1) = y;
        bigRow.at<float>(0, 2) = w;
        bigRow.at<float>(0, 3) = h;

        for (int k = 4; k < 14; k += 2) {
            bigRow.at<float>(0, k)     = smallRow.at<float>(0, k)     / scale;
            bigRow.at<float>(0, k + 1) = smallRow.at<float>(0, k + 1) / scale;
        }

        MatchResult r;
        if (!matcher_.best(encoder_.encode(frame, bigRow), r))
            continue;

        std::ostringstream label;
        label << std::fixed << std::setprecision(2);
        if (r.isMatch)
            label << "MATCH: " << nameFromPath(r.path) << " (" << r.score << ")";
        else
            label << "unknown (" << r.score << ")";

        result.push_back({box, label.str(), r.isMatch});
    }
    return result;
}

void VideoMatcher::draw(cv::Mat& frame) const
{
    for (const auto& v : views_)
    {
        cv::Scalar color = v.match ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 200);
        cv::rectangle(frame, v.box, color, 2);
        int y = v.box.y > 25 ? v.box.y - 8 : v.box.y + v.box.height + 20;
        cv::putText(frame, v.label, {v.box.x, y},
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 200, 0), 3);
    }
}

void VideoMatcher::run(const std::string& source)
{
    cv::VideoCapture cap;
    bool isIndex = !source.empty() && std::all_of(source.begin(), source.end(), ::isdigit);
    if (isIndex)
        cap.open(std::stoi(source));
    else    
        cap.open(source);
    if (!cap.isOpened())
        throw std::runtime_error("Не удалось открыть источник видео: " + source);

    cv::Mat frame;
    long frameIdx = 0;
    auto prev = std::chrono::steady_clock::now();

    while (cap.read(frame))
    {
        if (frameIdx % recomputeEvery_ == 0)
        {
            views_ = analyze(frame);
            bool anyMatch = std::any_of(views_.begin(), views_.end(),
                                        [](const FaceView& v) { return v.match; });
            streak_ = anyMatch ? streak_ + 1 : 0;
        }
        ++frameIdx;
        draw(frame);
        
        auto now = std::chrono::steady_clock::now();
        double fps = 1.0 / std::chrono::duration<double>(now - prev).count();
        prev = now;
        cv::putText(frame, "FPS: " + std::to_string(int(fps)),
                    {20, frame.rows - 20}, cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(255, 255, 255), 2);

        cv::imshow("Face Matcher", frame);
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q')
            break;
    }
}