#ifndef BLOCK_REPLACE_MANIPULATOR_H
#define BLOCK_REPLACE_MANIPULATOR_H
#include "FrameManipulator.h"
#include "DistanceMetric.h"
#include <vector>
#include <thread>

class BlockReplaceManipulator : public FrameManipulator
{
private:
  std::vector<cv::Mat> scaledImages;
  const DistanceMetric &distanceMetric;
  int blockWidth, blockHeight;

  // Helper function to scale images to match frame size
  std::vector<cv::Mat> scaleImages(const std::vector<cv::Mat> &images, const cv::Size &frameSize)
  {
    std::vector<cv::Mat> scaled;
    for (const auto &img : images)
    {
      if (img.empty())
      {
        throw std::runtime_error("One of the images is empty. Check input files!");
      }
      cv::Mat resized;
      cv::resize(img, resized, frameSize);
      scaled.push_back(resized);
    }
    return scaled;
  }

public:
  // Constructor
  BlockReplaceManipulator(const std::vector<cv::Mat> &images, const DistanceMetric &metric, int width, int height, const cv::Size &frameSize)
      : distanceMetric(metric), blockWidth(width), blockHeight(height)
  {
    if (images.empty())
    {
      throw std::invalid_argument("Image list cannot be empty!");
    }
    if (frameSize.width <= 0 || frameSize.height <= 0)
    {
      throw std::invalid_argument("Frame size must be positive!");
    }
    scaledImages = scaleImages(images, frameSize);
  }

  // Apply manipulation
  cv::Mat apply(const cv::Mat &frame) const override
  {
    cv::Mat output = frame.clone();
    cv::Size frameSize = frame.size();

    // Number of threads
    // std::thread::hardware_concurrency();
    const int numThreads = 8;
    std::vector<std::thread> threads;

    // Function to process a subset of rows
    auto processRows = [&](int startRow, int endRow)
    {
      for (int y = startRow; y < endRow; y += blockHeight)
      {
        for (int x = 0; x < frameSize.width; x += blockWidth)
        {
          int blockWidthClamped = std::min(blockWidth, frameSize.width - x);
          int blockHeightClamped = std::min(blockHeight, frameSize.height - y);
          cv::Rect blockRegion(x, y, blockWidthClamped, blockHeightClamped);

          cv::Mat block = frame(blockRegion).clone();

          double minDistance = std::numeric_limits<double>::max();
          cv::Mat bestMatch;

          for (const auto &img : scaledImages)
          {
            cv::Mat candidateBlock = img(blockRegion).clone();
            double distance = distanceMetric.calculate(block, candidateBlock);

            if (distance < minDistance)
            {
              minDistance = distance;
              bestMatch = candidateBlock.clone();
            }
          }

          if (!bestMatch.empty())
          {
            bestMatch.copyTo(output(blockRegion));
          }
        }
      }
    };

    // Divide the frame into rows for each thread
    int rowsPerThread = frameSize.height / numThreads;
    for (int i = 0; i < numThreads; ++i)
    {
      int startRow = i * rowsPerThread;
      int endRow = (i == numThreads - 1) ? frameSize.height : (i + 1) * rowsPerThread;
      threads.emplace_back(processRows, startRow, endRow);
    }

    // Join threads
    for (auto &thread : threads)
    {
      thread.join();
    }

    return output;
  }
};

#endif // BLOCK_REPLACE_MANIPULATOR_H
