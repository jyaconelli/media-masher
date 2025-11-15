#ifndef GRAYSCALE_MANIPULATOR_H
#define GRAYSCALE_MANIPULATOR_H

#include "FrameManipulator.h"

class GrayscaleManipulator : public FrameManipulator
{
public:
  cv::Mat apply(const cv::Mat &frame) const override
  {
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    return grayFrame;
  }
};

#endif // GRAYSCALE_MANIPULATOR_H
