#pragma once
#include <vector>
#include <queue>
#include <limits>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "shape.hpp"
#include "gridmap.hpp"

/**
 * A* 搜索空间：离散 SE(2) -> (x, y, theta_idx)
 * - 邻接动作：直行、左/右小转+前进、原地左/右小转
 * - 碰撞检测：调用 PolygonalShape::collidesAt
 * - 只输出最终路径（不显示展开过程）
 */

struct Node {
  int x{0}, y{0}, th{0};
  double g{std::numeric_limits<double>::infinity()};
  double f{std::numeric_limits<double>::infinity()};
  int px{-1}, py{-1}, pth{-1};  // parent
  bool closed{false};
  bool opened{false};
};

struct AnyShapeAStar {
  const GridMap& gm;
  const PolygonalShape& shape;
  int    theta_bins{36};
  double step_pix{5.0};      // 机器人更大 -> 步长略大
  double yaw_step_bins{1};
  double w_yaw{0.010};        // 角度权重
  bool   visualize_expand{false}; // 本工程仅最终路径显示

  AnyShapeAStar(const GridMap& map, const PolygonalShape& sh)
    : gm(map), shape(sh) {}

  inline double yawOf(int th) const {
    const double d = (2.0*M_PI)/theta_bins;
    return d * th;
  }

  struct PQItem { double f; int idx; bool operator<(const PQItem& o) const { return f > o.f; } };
  inline int idx(int x,int y,int th) const { return (th*gm.H + y)*gm.W + x; }

  inline bool collides(int x, int y, int th) const {
    return shape.collidesAt(gm.occ, x, y, yawOf(th), gm.meter_per_pixel, nullptr);
  }

  inline double heuristic(int x,int y,int gx,int gy,int th,int gth) const {
    double dx=x-gx, dy=y-gy;
    double d = std::sqrt(dx*dx+dy*dy);
    double yaw = yawOf(th), gyaw = yawOf(gth);
    double dth = std::atan2(std::sin(yaw-gyaw), std::cos(yaw-gyaw));
    return d + w_yaw*std::abs(dth)*(180.0/M_PI);
  }

  std::vector<Eigen::Vector3i> plan(
      Eigen::Vector3i start,
      Eigen::Vector3i goal)
  {
    // 起点若碰撞，直接失败
    if (collides(start.x(), start.y(), start.z()))
      return {};

    const int N = gm.W*gm.H*theta_bins;
    std::vector<Node> nodes(N);

    auto recPath = [&](int id){
      std::vector<Eigen::Vector3i> path;
      while(id!=-1){
        const Node& n = nodes[id];
        path.emplace_back(n.x, n.y, n.th);
        id = (n.px<0)? -1 : idx(n.px,n.py,n.pth);
      }
      std::reverse(path.begin(), path.end());
      return path;
    };

    auto pushOrRelax = [&](int nx,int ny,int nth,double ng,int px,int py,int pth,
                           std::priority_queue<PQItem>& pq){
      if(!gm.inBounds(nx,ny)) return;
      int id = idx(nx,ny,nth);
      if (nodes[id].closed) return;
      if (collides(nx,ny,nth)) return;

      double nf = ng + heuristic(nx,ny,goal.x(),goal.y(),nth,goal.z());
      if (!nodes[id].opened || ng < nodes[id].g) {
        nodes[id].opened = true;
        nodes[id].g = ng; nodes[id].f = nf;
        nodes[id].x=nx; nodes[id].y=ny; nodes[id].th=nth;
        nodes[id].px=px; nodes[id].py=py; nodes[id].pth=pth;
        pq.push({nf, id});
      }
    };

    // 初始化
    std::priority_queue<PQItem> pq;
    const int sid = idx(start.x(),start.y(),start.z());
    nodes[sid].opened=true;
    nodes[sid].g=0.0;
    nodes[sid].f=heuristic(start.x(),start.y(),goal.x(),goal.y(),start.z(),goal.z());
    nodes[sid].x=start.x(); nodes[sid].y=start.y(); nodes[sid].th=start.z();
    pq.push({nodes[sid].f, sid});

    // 五种邻居动作
    const std::array<int,5> dths = {0, +1, -1, +1, -1};
    const std::array<double,5> fwd = {step_pix, step_pix, step_pix, 0.0, 0.0};

    while(!pq.empty()){
      const int id = pq.top().idx; pq.pop();
      Node& cur = nodes[id];
      if (cur.closed) continue;
      cur.closed = true;

      // 抵达判定：靠近位置 & 角度接近
      double dx=cur.x-goal.x(), dy=cur.y-goal.y();
      if (std::sqrt(dx*dx+dy*dy) <= step_pix*1.5 &&
          (std::abs(cur.th-goal.z()) % theta_bins) <= 1) {
        return recPath(id);
      }

      // 扩展动作
      for (int k=0;k<5;++k){
        int nth = ( (cur.th + (int)std::round(dths[k]*yaw_step_bins)) % theta_bins + theta_bins ) % theta_bins;
        double yaw = yawOf(nth);
        int nx = cur.x + static_cast<int>(std::round(fwd[k]*std::cos(yaw)));
        int ny = cur.y + static_cast<int>(std::round(fwd[k]*std::sin(yaw)));
        double travel = fwd[k];
        double yaw_pen = (k>=3 ? 1.0 : 0.1); // 原地转向稍贵
        double ng = cur.g + travel + yaw_pen*w_yaw;
        pushOrRelax(nx,ny,nth,ng, cur.x,cur.y,cur.th, pq);
      }
    }
    return {}; // 无解
  }
};
