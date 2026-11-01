#pragma once
#include <opencv2/opencv.hpp>
#include <random>

/**
 * GridMap：单通道 8U 占据栅格（0=free, 255=occupied）
 * 仅用于演示，提供随机矩形/圆形/小点障碍。
 */
struct GridMap {
  int W{800}, H{500};              // 像素尺寸（放大，方便更大机器人活动）
  double meter_per_pixel{0.05};    // 5 cm / pixel
  cv::Mat occ;                     // 占据图
  cv::Mat dist;                    // 距离场：每个空闲像素到最近障碍的 L2 距离（像素）

  GridMap(int w, int h, double mpp)
    : W(w), H(h), meter_per_pixel(mpp), occ(h, w, CV_8UC1, cv::Scalar(0)) {}

  // 随机矩形块
  void addRandomBlocks(int n, int minw=20, int minh=14, int maxw=110, int maxh=70, unsigned seed=1)
  {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<> xdist(0, W-1), ydist(0, H-1);
    std::uniform_int_distribution<> wdist(minw, maxw), hdist(minh, maxh);
    for (int i=0;i<n;++i) {
      int w = std::min(wdist(rng), W-1);
      int h = std::min(hdist(rng), H-1);
      int x = std::min(xdist(rng), W-w-1);
      int y = std::min(ydist(rng), H-h-1);
      cv::rectangle(occ, cv::Rect(x,y,w,h), cv::Scalar(255), cv::FILLED);
    }
  }

  // 指定矩形障碍（像素坐标），用于手工拼出 L 形墙体等
  void addRect(int x, int y, int w, int h) {
    cv::rectangle(occ, cv::Rect(x, y, w, h), cv::Scalar(255), cv::FILLED);
  }

  // 随机圆形块
  void addRandomCircles(int n, int rmin=9, int rmax=26, unsigned seed=2) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<> xdist(0, W-1), ydist(0, H-1);
    std::uniform_int_distribution<> rdist(rmin, rmax);
    for (int i=0;i<n;++i) {
      cv::circle(occ, {xdist(rng), ydist(rng)}, rdist(rng), cv::Scalar(255), cv::FILLED);
    }
  }

  // 大量“零星小点”障碍（半径 1~2 像素）
  void addRandomDots(int n, unsigned seed=3) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<> xdist(0, W-1), ydist(0, H-1);
    std::uniform_int_distribution<> rdist(1, 2);
    for (int i=0;i<n;++i) {
      cv::circle(occ, {xdist(rng), ydist(rng)}, rdist(rng), cv::Scalar(255), cv::FILLED);
    }
  }

  // 预计算距离场（障碍搭建完成后调用一次）：用于“远离障碍”软惩罚
  void computeDistanceField() {
    cv::Mat bin;  // 障碍=0，空闲=255
    cv::threshold(occ, bin, 127, 255, cv::THRESH_BINARY_INV);
    cv::distanceTransform(bin, dist, cv::DIST_L2, 3);
  }

  inline bool inBounds(int x, int y) const { return (x>=0 && y>=0 && x<W && y<H); }
};
