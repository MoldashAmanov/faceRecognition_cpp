#include "FaceDetector.hpp"
#include "FaceEncoder.hpp"
#include "Gallery.hpp"
#include "FaceMatcher.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Использование: faceRecognition <query.jpg> <папка галереи> [результат.jpg]\n";
        return 1;
    }

    const std::string queryPath = argv[1], galleryDir = argv[2];
    const std::string outPath = argc > 3 ? argv[3] : "result.jpg";

    try 
    {
        FaceDetector detector("models/face_detection_yunet_2023mar.onnx");
        FaceEncoder encoder("models/face_recognition_sface_2021dec.onnx");

        cv::Mat query = cv::imread(queryPath);
        if (query.empty())
            throw std::runtime_error("Не удалось открыть " + queryPath);

        cv::Mat queryFace;
        if (!detector.detectBest(query, queryFace))
        {
            std::cout << "На запросном фото лицо не найдено\n";
            return 2;
        }
        cv::Mat queryFeature = encoder.encode(query, queryFace);

        Gallery gallery(detector, encoder);
        size_t n = gallery.load(galleryDir);
        std::cout << "В галереи лиц: " << n << ", пропущено файлов: " << gallery.skipped().size() << "\n\n";

        FaceMatcher matcher(encoder, gallery);
        auto results = matcher.match(queryFeature);

        std::cout << std::fixed << std::setprecision(3);
        for (const auto& r : results)
        {
            std::cout << (r.isMatch ? "[СОВПАДЕНИЕ ]" : "[нет]       ") << r.score << "   " << r.path << "\n";
        }
        bool found = !results.empty() && results.front().isMatch;
        cv::Rect box(queryFace.at<float>(0, 0), queryFace.at<float>(0, 1),
                    queryFace.at<float>(0, 2), queryFace.at<float>(0, 3));
        cv::rectangle(query, box, found ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 220), 2);
        cv::imwrite(outPath, query);
        std::cout << "\nРезультат сохранен в " << outPath << "\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Ошибка: " << e.what() << "\n";
        return 3;
    }
    return 0;
}