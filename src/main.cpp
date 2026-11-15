#include "FaceDetector.hpp"
#include "FaceEncoder.hpp"
#include "Gallery.hpp"
#include "FaceMatcher.hpp"
#include "VideoMatcher.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iomanip>
#include <iostream>


namespace {
const char* kDetModel = "models/face_detection_yunet_2023mar.onnx";
const char* kRecModel = "models/face_recognition_sface_2021dec.onnx";

void printUsage()
{
     std::cout << "Использование:\n" 
        << "  face_matcher photo <query.jpg> <папка_галереи> [result.jpg]\n"
        << "  face_matcher vidoe <папка_галереи> [камера|видеофайл]\n";
}

int runPhoto(FaceDetector& detector, FaceEncoder& encoder, const std::string& queryPath,
            const std::string& galleryDir, const std::string& outPath)
{
    cv::Mat query = cv::imread(queryPath);
    if (query.empty())
        throw std::runtime_error("Не удалось открыть: " + queryPath);
    
    cv::Mat queryFace;
    if (!detector.detectBest(query, queryFace))
    {
        std::cout << "На запросном фото лицо не найдено\n";
        return 2;
    }
    cv::Mat feature = encoder.encode(query, queryFace);

    Gallery gallery(detector, encoder);
    size_t n = gallery.load(galleryDir);
    std::cout << "В галереи лиц: " << n << ", пропущено файлов: " << gallery.skipped().size() << "\n\n";

    FaceMatcher matcher(encoder, gallery);
    auto results = matcher.match(feature);

    std::cout << std::fixed << std::setprecision(3);
    for (const auto& r : results)
    {
        std::cout << (r.isMatch ? "[СОВПАДЕНИЕ]" : "[нет]       ") << r.score << "   " << r.path << "\n";
    }
    bool found = !results.empty() && results.front().isMatch;
    cv::Rect box(queryFace.at<float>(0, 0), queryFace.at<float>(0, 1),
                queryFace.at<float>(0, 2), queryFace.at<float>(0, 3));
    cv::rectangle(query, box, found ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 220), 2);
    cv::imwrite(outPath, query);
    std::cout << "\nРезультат сохранен в " << outPath << "\n";
    return 0;
}   

int runVideo(FaceDetector& detector, FaceEncoder& encoder,
            const std::string& galleryDir, const std::string& source)
{
    Gallery gallery(detector, encoder);
    size_t n = gallery.load(galleryDir);
    std::cout << "В галерее лиц: " << n << "\n";
    if (n == 0)
    {
        std::cerr << "Галерея пуста\n";
        return 2;
    }

    FaceMatcher matcher(encoder, gallery);
    VideoMatcher video(detector, encoder, matcher, 3, 3);
    video.run(source);
    return 0;
}
}   // namespace


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        printUsage();
        return 1;
    }
    const std::string mode = argv[1];

    try 
    {
        FaceDetector detector(kDetModel);
        FaceEncoder encoder(kRecModel);

        if (mode == "photo" && argc >= 4)
        {
            return runPhoto(detector, encoder, argv[2], argv[3], argc > 4 ? argv[4] : "result.jpg");
        }
        if (mode == "video")
        {
            return runVideo(detector, encoder, argv[2], argc > 3 ? argv[3] : "0");
        }

        printUsage();
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Ошибка: " << e.what() << "\n";
        return 3;
    }
}