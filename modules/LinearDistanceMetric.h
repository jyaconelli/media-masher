#ifndef LINEAR_DISTANCE_METRIC_H
#define LINEAR_DISTANCE_METRIC_H

#include "DistanceMetric.h"

class LinearDistanceMetric : public DistanceMetric
{
public:
  double calculate(const cv::Mat &block1, const cv::Mat &block2) const override
  {
    cv::Scalar sum1 = cv::sum(block1);
    cv::Scalar sum2 = cv::sum(block2);
    double total1 = sum1[0] + sum1[1] + sum1[2]; // Sum of R, G, B channels
    double total2 = sum2[0] + sum2[1] + sum2[2];
    return std::abs(total1 - total2); // Linear distance
  }
};

#endif // LINEAR_DISTANCE_METRIC_H
