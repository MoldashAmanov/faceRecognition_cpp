#include "FaceDetector.hpp"
#include "FaceEncoder.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Использование: faceRecognition <фото1> <фото2>\n";
        return 1;
    }
    try 
    {
        FaceDetector detector("models/face_detection_yunet_2023mar.onnx");
        FaceEncoder encoder("models/face_recognition_sface_2021dec.onnx");

        cv::Mat a = cv::imread(argv[1]), b = cv::imread(argv[2]);
        if (a.empty()) { std::cerr << "Не загрузилось: " << argv[1] << "\n"; return 1; }
        if (b.empty()) { std::cerr << "Не загрузилось: " << argv[2] << "\n"; return 1; }

        std::cout << argv[1] << ": " << a.cols << "x" << a.rows << "\n";
        std::cout << argv[2] << ": " << b.cols << "x" << b.rows << "\n";

        cv::Mat fa, fb;
        if (!detector.detectBest(a, fa)) { std::cerr << "Лицо не найдено: " << argv[1] << "\n"; return 1; }
        if (!detector.detectBest(b, fb)) { std::cerr << "Лицо не найдено: " << argv[2] << "\n"; return 1; }
    
    double s = encoder.similarity(encoder.encode(a, fa), encoder.encode(b, fb));
    std::cout << "Схожесть: " << s << (s >= 0.363 ? " -> один человек\n" : " -> разные люди\n");
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return 2;
    }
}