#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <unordered_map>
#include <mutex>
#include <cmath>
#include <CvPlot/cvplot.h>

extern "C" {
#include "btlemon.h"
}

#define MAX_HISTORY_LENGTH 1000
#define MA_FACTOR 30
#define MODE_LENGTH 200
#define PAUSE_LENGTH 50

std::unordered_map<std::string, int> addr_rssi;
std::mutex addr_rssi_mutex;

std::string addr_string(const uint8_t addr[6]) {
  std::stringstream ss;
  char hex[3];
  for (int i = 5; i >= 0; i--) {
    sprintf(hex, "%2.2X", addr[i]);
    ss << (i < 5 ? ":" : "") << hex;
  }
  return ss.str();
}

void callback(const uint8_t addr[6], const int8_t *rssi) {
  const std::lock_guard<std::mutex> lock(addr_rssi_mutex);
  addr_rssi[addr_string(addr)] = *rssi;
}

double rssi_distance(double rssi, double tx, double n) {
  return std::pow(10, (tx - rssi) / n);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "ERROR: pass target device address as first argument, e.g. 78:AA:A7:94:0A:31\n";
    return -1;
  }

  cv::Ptr<cv::VideoWriter> writerp;

  if (argc == 3) {
    writerp = new cv::VideoWriter(argv[2], CV_FOURCC('M','J','P','G'), 18, {1280, 720});
  }

  double tx = -61.02;
  double n = 21.87;
  double current_rssi = 0;
  double current_distance = 0;
  const std::string target_device_addr(argv[1]);
  std::unordered_map<std::string, std::vector<int>> addr_rssi_history;
  std::vector<double> ema_history;
  std::vector<double> dist_history;

  std::vector<std::pair<std::string, int>> modes = {
      {"", PAUSE_LENGTH},
      {"front pocket, screen facing receiver", MODE_LENGTH},
      {"", PAUSE_LENGTH},
      {"front pocket, back facing receiver", MODE_LENGTH},
      {"", PAUSE_LENGTH},
      {" back pocket, screen facing receiver", MODE_LENGTH},
      {"", PAUSE_LENGTH},
      {" back pocket, back facing receiver", MODE_LENGTH},
      {"", PAUSE_LENGTH},
      {"     in hand, back facing receiver", MODE_LENGTH},
      {"", PAUSE_LENGTH}
  };

  int current_mode = 0;
  int current_frame = 0;
  int next_mode_at = modes[0].second;
  int total_video_length = 0;
  for (auto& mode: modes) {
    total_video_length += mode.second;
  }

  btlemon_set_callback(callback);
  std::thread thread(btlemon_run);

  cv::Mat camera_frame;
  cv::Mat video_frame(720, 1280, CV_8UC3);
  cv::Rect camera_frame_roi(0, 120, 640, 480);
  cv::Rect plot_frame_roi(640, 0, 640, 360);
  cv::Rect plot2_frame_roi(640, 360, 640, 360);
  cv::VideoCapture cap;
  cap.open(0, cv::CAP_ANY);
  if (!cap.isOpened()) {
    std::cerr << "ERROR: Failed to open camera\n";
    return -1;
  }
  cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
  for (;;) {
    video_frame.setTo(0);
    cap.read(camera_frame);
    if (camera_frame.empty()) {
      std::cerr << "ERROR: Frame empty\n";
      break;
    }
    camera_frame.copyTo(video_frame(camera_frame_roi));

    if (ema_history.size() > MAX_HISTORY_LENGTH) {
      ema_history.erase(ema_history.begin());
    }
    if (dist_history.size() > MAX_HISTORY_LENGTH) {
      dist_history.erase(dist_history.begin());
    }
    {
      const std::lock_guard<std::mutex> lock(addr_rssi_mutex);
      for (auto& it: addr_rssi) {
        std::vector<int>& history = addr_rssi_history[it.first];
        history.emplace_back(it.second);
        if (history.size() > MAX_HISTORY_LENGTH) {
          history.erase(history.begin());
        }
        if (it.first == target_device_addr) {
          if (history.size() > 10 && ema_history.empty()) {
            double avg = 0;
            for (int entry : history)
              avg += entry;
            avg /= history.size();
            ema_history.push_back(avg);
            dist_history.push_back(rssi_distance(avg, tx, n));
          }
          double avg_distance = MA_FACTOR;
          if (!ema_history.empty()) {
            double avg = ema_history.back();
            avg -= avg / avg_distance;
            avg += it.second / avg_distance;
            ema_history.push_back(avg);
            current_rssi = avg;
            current_distance = rssi_distance(avg, tx, n);
            dist_history.push_back(current_distance);
          }
        }
      }
    }

    if (addr_rssi_history.find(target_device_addr) != addr_rssi_history.end()) {
      auto axes = CvPlot::makePlotAxes();
      axes.create<CvPlot::Series>(addr_rssi_history[target_device_addr], "-r");
      axes.create<CvPlot::Series>(ema_history, "-b");
      axes.setYLim({-100, 0});
      axes.setXLim({0, MAX_HISTORY_LENGTH});
      axes.title("RSSI");
      cv::Mat plot_frame = axes.render(360, 640);
      plot_frame.copyTo(video_frame(plot_frame_roi));

      auto axes2 = CvPlot::makePlotAxes();
      axes2.create<CvPlot::Series>(dist_history, "-b");
      axes2.setYLim({0, 10});
      axes2.setXLim({0, MAX_HISTORY_LENGTH});
      axes2.title("distance [m]");
      cv::Mat plot_frame2 = axes2.render(360, 640);
      plot_frame2.copyTo(video_frame(plot2_frame_roi));
    }

    std::stringstream top_ss;
    top_ss << std::setprecision(2) << "RSSI: "<< current_rssi << ", measured distance: " << current_distance << " m, true distance: ~1 m";

    std::string bottom_s;
    if (current_mode < modes.size()) {
      bottom_s = modes[current_mode].first;
      if (current_frame > next_mode_at) {
        next_mode_at += modes[++current_mode].second;
      }
    }

    cv::putText(video_frame, top_ss.str(), {15,100}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255});
    cv::putText(video_frame, bottom_s, {15, 625}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255});
    cv::imshow("btlemon", video_frame);

    if (writerp) {
      writerp->write(video_frame);
    }

    if (cv::waitKey(10) == 27) {
      break;
    }

    current_frame++;
    if (writerp && current_frame > total_video_length) {
      break;
    }
  }
  std::cout << "Terminating btlemon..." << std::flush;
  btlemon_stop();
  thread.join();
  std::cout << " done\n";
  return 0;
}