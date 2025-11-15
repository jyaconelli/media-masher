#include <opencv2/opencv.hpp>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include "BlockReplaceManipulator.h"
#include "LinearDistanceMetric.h"
#include <chrono>

// Comparator for priority queue: Compare only the int part of the pair
auto frameComparator = [](const std::pair<int, cv::Mat> &a, const std::pair<int, cv::Mat> &b)
{
  return a.first > b.first; // Min-heap based on frame index
};

// Maximum size for producer queue to limit memory usage
constexpr size_t MAX_QUEUE_SIZE = 10;

// Thread-safe priority queue for frames with custom comparator
std::priority_queue<std::pair<int, cv::Mat>, std::vector<std::pair<int, cv::Mat>>, decltype(frameComparator)> frameQueue(frameComparator);
std::mutex queueMutex;
std::condition_variable queueCondition;

// Thread-safe output queue for sequential writing
std::priority_queue<std::pair<int, cv::Mat>, std::vector<std::pair<int, cv::Mat>>, decltype(frameComparator)> outputQueue(frameComparator);
std::mutex outputMutex;
std::condition_variable outputCondition;

// Producer: Reads frames and adds them to the queue
void frameProducer(cv::VideoCapture &cap, int &frameIndex, bool &finished)
{
  cv::Mat frame;
  while (cap.read(frame))
  {
    {
      std::unique_lock<std::mutex> lock(queueMutex);
      std::cout << "[Producer] Lock acquired for frameQueue. Checking condition..." << std::endl;
      queueCondition.wait(lock, []
                          { return frameQueue.size() < MAX_QUEUE_SIZE; });

      frameQueue.emplace(frameIndex++, frame.clone());
      std::cout << "[Producer] Produced frame " << frameIndex << " Queue size: " << frameQueue.size() << std::endl;
    }
    std::cout << "[Producer] Lock released for frameQueue. Notifying consumers..." << std::endl;
    queueCondition.notify_all();
  }
  finished = true;
  std::cout << "[Producer] Finished producing frames. Notifying consumers..." << std::endl;
  queueCondition.notify_all();
}

// Consumer: Processes frames from the queue and pushes them to the output queue
void frameConsumer(BlockReplaceManipulator &manipulator, bool &finished)
{
  while (true)
  {
    std::pair<int, cv::Mat> item;

    {
      std::unique_lock<std::mutex> lock(queueMutex);
      std::cout << "[Consumer] Lock acquired for frameQueue. Checking condition..." << std::endl;
      queueCondition.wait(lock, [&]()
                          { return !frameQueue.empty() || finished; });

      if (frameQueue.empty() && finished)
      {
        std::cout << "[Consumer] No frames left and production finished. Exiting." << std::endl;
        break;
      }

      item = frameQueue.top();
      frameQueue.pop();
      std::cout << "[Consumer] Consumed frame " << item.first << " Queue size: " << frameQueue.size() << std::endl;
    }

    cv::Mat manipulatedFrame;
    try
    {
      auto start = std::chrono::high_resolution_clock::now();
      manipulatedFrame = manipulator.apply(item.second);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      std::cout << "[Consumer] Frame " << item.first << " manipulated in " << elapsed.count() << " seconds" << std::endl;
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error applying manipulator: " << e.what() << std::endl;
      manipulatedFrame = item.second; // Pass through the original frame
    }

    {
      std::lock_guard<std::mutex> lock(outputMutex);
      outputQueue.emplace(item.first, manipulatedFrame);
      std::cout << "[Consumer] Produced manipulated frame " << item.first << " Output queue size: " << outputQueue.size() << std::endl;
      outputCondition.notify_all();
    }
  }
}

// Writer: Writes frames from the output queue in order
void outputWriter(cv::VideoWriter &writer, bool &finished)
{
  int nextFrameIndex = 0;

  while (true)
  {
    std::pair<int, cv::Mat> item;

    {
      std::unique_lock<std::mutex> lock(outputMutex);
      std::cout << "[Writer] Lock acquired for outputQueue. Checking condition..." << std::endl;
      outputCondition.wait(lock, [&]()
                           { return !outputQueue.empty() || finished; });

      if (outputQueue.empty() && finished)
      {
        std::cout << "[Writer] No frames left and production finished. Exiting." << std::endl;
        break;
      }

      item = outputQueue.top();
      if (item.first != nextFrameIndex)
      {
        std::cout << "[Writer] Waiting for frame " << nextFrameIndex << ". Current frame: " << item.first << std::endl;
        outputCondition.wait(lock, [&]()
                             { return !outputQueue.empty() && outputQueue.top().first == nextFrameIndex; });
        continue; // Ensure frames are written in order
      }

      outputQueue.pop();
      std::cout << "[Writer] Writing frame " << item.first << " Output queue size: " << outputQueue.size() << std::endl;
      ++nextFrameIndex;
    }

    try
    {
      writer.write(item.second);
    }
    catch (const cv::Exception &e)
    {
      std::cerr << "Error writing frame: " << e.what() << std::endl;
    }
  }
}

int main(int argc, char *argv[])
{
  // Check arguments
  if (argc < 6)
  {
    std::cerr << "Usage: " << argv[0]
              << " <video_file_path> <image_list_comma_separated> <output_file_path> <block_width> <block_height>"
              << std::endl;
    return -1;
  }

  std::string videoFilePath = argv[1];
  std::string imageList = argv[2];
  std::string outputFilePath = argv[3];
  int blockWidth = std::stoi(argv[4]);
  int blockHeight = std::stoi(argv[5]);

  // Parse image paths
  std::vector<std::string> imagePaths;
  std::stringstream ss(imageList);
  std::string imagePath;
  while (std::getline(ss, imagePath, ','))
  {
    imagePaths.push_back(imagePath);
  }

  // Load images
  std::vector<cv::Mat> images;
  for (const auto &path : imagePaths)
  {
    cv::Mat img = cv::imread(path);
    if (img.empty())
    {
      std::cerr << "Error: Could not load image: " << path << std::endl;
      return -1;
    }
    images.push_back(img);
  }

  // Open video file
  cv::VideoCapture cap(videoFilePath);
  if (!cap.isOpened())
  {
    std::cerr << "Error: Could not open video file: " << videoFilePath << std::endl;
    return -1;
  }

  // Get video properties
  int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
  int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

  // Create video writer
  cv::VideoWriter writer(outputFilePath, fourcc, fps, cv::Size(frameWidth, frameHeight), true);
  if (!writer.isOpened())
  {
    std::cerr << "Error: Could not open output file: " << outputFilePath << std::endl;
    return -1;
  }

  // Initialize manipulator
  cv::Size frameSize(frameWidth, frameHeight);
  LinearDistanceMetric distanceMetric;
  BlockReplaceManipulator blockReplaceManipulator(images, distanceMetric, blockWidth, blockHeight, frameSize);

  // Multithreaded frame processing
  int frameIndex = 0;
  bool finished = false;

  // Start producer thread
  std::thread producer(frameProducer, std::ref(cap), std::ref(frameIndex), std::ref(finished));

  // Start consumer threads
  std::vector<std::thread> consumers;
  int numConsumers = std::thread::hardware_concurrency();
  for (int i = 0; i < numConsumers; ++i)
  {
    consumers.emplace_back(frameConsumer, std::ref(blockReplaceManipulator), std::ref(finished));
  }

  // Start writer thread
  std::thread writerThread(outputWriter, std::ref(writer), std::ref(finished));

  producer.join();
  for (auto &consumer : consumers)
  {
    consumer.join();
  }

  {
    std::lock_guard<std::mutex> lock(outputMutex);
    finished = true;
    std::cout << "[Main] All threads finished. Notifying writer..." << std::endl;
    outputCondition.notify_all();
  }
  writerThread.join();

  cap.release();
  writer.release();

  std::cout << "Processing complete. Output saved to: " << outputFilePath << std::endl;

  return 0;
}
