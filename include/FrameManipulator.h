#ifndef FRAME_MANIPULATOR_H
#define FRAME_MANIPULATOR_H

#include <opencv2/opencv.hpp>

// Abstract class for frame manipulation
class FrameManipulator
{
public:
  virtual ~FrameManipulator() = default;

  // Apply manipulation to a frame
  virtual cv::Mat apply(const cv::Mat &frame) const = 0;
};

#endif // FRAME_MANIPULATOR_H
