#ifndef DISTANCE_METRIC_H
#define DISTANCE_METRIC_H

#include <opencv2/opencv.hpp>

// Interface for a distance metric
class DistanceMetric
{
public:
  virtual ~DistanceMetric() = default;

  // Calculate distance between two pixel blocks
  virtual double calculate(const cv::Mat &block1, const cv::Mat &block2) const = 0;
};

#endif // DISTANCE_METRIC_H
