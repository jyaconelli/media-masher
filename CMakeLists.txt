cmake_minimum_required(VERSION 3.10)
project(video_manipulator)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add include directories for header files
include_directories(include)
include_directories(modules)

# Add source files
add_executable(video_manipulator
    src/main.cpp
    modules/GrayscaleManipulator.h
)

# Link OpenCV libraries
target_link_libraries(video_manipulator ${OpenCV_LIBS})
