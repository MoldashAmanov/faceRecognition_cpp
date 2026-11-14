cmake_minimum_required(VERSION 3.16)
project(faceRecognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")


find_package(OpenCV REQUIRED COMPONENTS
    core imgproc imgcodecs videoio dnn objdetect highgui
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")


set(SOURCES
    src/main.cpp
    src/FaceDetector.cpp

)

add_executable(faceRecognition_cpp ${SOURCES})


target_include_directories(faceRecognition_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(faceRecognition_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(faceRecognition_cpp PRIVATE ${OpenCV_LIBS})