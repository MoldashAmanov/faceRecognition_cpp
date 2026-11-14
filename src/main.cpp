#include "FaceDetector.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Использование: faceRecognition <фото>\n";
        return 1;
    }
    try 
    {
        FaceDetector detector("models/face_detection_yunet_2023mar.onnx");
        cv::Mat img = cv::imread(argv[1]);
        if (img.empty())
        {
            std::cerr << "Нет файла\n";
            return 1;
        }
        cv::Mat faces = detector.detect(img);
        std::cout << "Найдено лиц: " << faces.rows << "\n";

        for (int i = 0; i < faces.rows; ++i)
        {
            cv::Rect box(faces.at<float>(i, 0), faces.at<float>(i, 1),
                        faces.at<float>(i, 2), faces.at<float>(i, 3));
            cv::rectangle(img, box, {0, 200, 0}, 2);
            for (int p = 0; p < 5; ++p) // 5 ключевых точек
            {
                cv::circle(img, {int(faces.at<float>(i, 4 + 2 * p)),
                                int(faces.at<float>(i, 5 + 2 * p))}, 3, {0, 0, 255}, -1);
            }
        }

        cv::imshow("Detect", img);
        cv::waitKey(0);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return 2;
    }
    
}