#pragma once
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

/**
 * PolygonalShape
 *  - 描述一个“外轮廓 + 若干内孔(holes)”的多边形形状，单位：米（形心在(0,0)）。
 *  - 提供：按 yaw 旋转 -> 栅格坐标 -> 在地图 ROI 内做掩膜重叠的碰撞检测。
 *  - 用于前端 A* 的任意形状碰撞。
 */
class PolygonalShape {
public:
  // 外轮廓（逆时针/顺时针均可，建议逆时针）
  std::vector<Eigen::Vector2d> outer_m_;
  // 多个内孔（任意方向）
  std::vector<std::vector<Eigen::Vector2d>> holes_m_;

  // 将形状顶点（米）按 yaw 旋转并换算为像素坐标（以形心为原点）
  static std::vector<cv::Point> rotateToPixels(
      const std::vector<Eigen::Vector2d>& verts_m,
      double yaw_rad, double meter_per_pixel)
  {
    const double c = std::cos(yaw_rad), s = std::sin(yaw_rad);
    std::vector<cv::Point> pts; pts.reserve(verts_m.size());
    for (auto& v : verts_m) {
      const double xr = c*v.x() - s*v.y();
      const double yr = s*v.x() + c*v.y();
      pts.emplace_back(
          static_cast<int>(std::round(xr / meter_per_pixel)),
          static_cast<int>(std::round(yr / meter_per_pixel)));
    }
    return pts;
  }

  /**
   * 碰撞检测：把形状以 (cx,cy) 为形心、yaw_rad 朝向放置到 map_bin 上，检测与障碍是否重叠
   * @param map_bin  单通道 8U：0=空闲，>0=障碍
   * @return true 表示碰撞；false 表示无碰撞
   */
  bool collidesAt(const cv::Mat& map_bin, int cx, int cy,
                  double yaw_rad, double meter_per_pixel,
                  cv::Rect* out_roi_debug = nullptr) const
  {
    // 1) 旋转到像素坐标（形心为原点）
    auto outer_pix = rotateToPixels(outer_m_, yaw_rad, meter_per_pixel);
    std::vector<std::vector<cv::Point>> holes_pix;
    holes_pix.reserve(holes_m_.size());
    for (auto& hole : holes_m_) {
      holes_pix.emplace_back(rotateToPixels(hole, yaw_rad, meter_per_pixel));
    }

    // 2) 求所有多边形的包围盒，并平移到 (cx,cy)
    cv::Rect bbox = cv::boundingRect(outer_pix);
    for (auto& hp : holes_pix) bbox |= cv::boundingRect(hp);
    cv::Rect roi(bbox.x + cx, bbox.y + cy, bbox.width, bbox.height);

    // 越界直接视作碰撞（也可裁剪处理）
    if (roi.x < 0 || roi.y < 0 ||
        roi.x + roi.width > map_bin.cols ||
        roi.y + roi.height > map_bin.rows)
      return true;

    // 3) 在 ROI 内绘制外轮廓填充，再用 0 抠出内孔
    cv::Mat shapeMask = cv::Mat::zeros(roi.height, roi.width, CV_8UC1);

    auto shiftToROI = [&](std::vector<cv::Point>& poly){
      for (auto& p : poly) { p.x -= bbox.x; p.y -= bbox.y; }
    };
    shiftToROI(outer_pix);
    for (auto& hp : holes_pix) shiftToROI(hp);

    const std::vector<std::vector<cv::Point>> outv{outer_pix};
    cv::fillPoly(shapeMask, outv, cv::Scalar(255));
    for (auto& hp : holes_pix) {
      const std::vector<std::vector<cv::Point>> holev{hp};
      cv::fillPoly(shapeMask, holev, cv::Scalar(0));
    }

    // 4) 与地图 ROI 做按位与，若重叠非零则碰撞
    cv::Mat overlap;
    cv::bitwise_and(shapeMask, map_bin(roi), overlap);
    if (out_roi_debug) *out_roi_debug = roi;
    return cv::countNonZero(overlap) > 0;
  }

  // ------------------- 工具：构造 F / A 放大形状（单位：米） -------------------
// 放大且更规范的 "F" 外轮廓：宽 w、高 h、笔画厚度 t
// 额外可选参数：mid_rel 控制中横梁相对顶边的位置（0~1，越大越靠下）；
//              mid_len_rel 控制中横梁相对全宽的长度比例（0~1）。
static PolygonalShape MakeF(double w, double h, double t,
                            double mid_rel = 0.40,      // 中横梁位置（相对 h）
                            double mid_len_rel = 0.60)  // 中横梁长度（相对 w）
{
  // 基本边界
  const double xL = -w/2.0, xR = +w/2.0;
  const double yB = -h/2.0, yT = +h/2.0;

  // 竖干右缘（竖干厚度 = t）
  const double xStemR = xL + t;

  // 顶横梁：厚度 t，长度全宽 [xL, xR]，竖向范围 [yT - t, yT]
  const double yTopBot = yT - t;  // 顶横梁下缘

  // 中横梁：中心高度 yMid，厚度 t，长度从竖干右缘到 xMidR
  const double yMid     = yT - h * mid_rel;
  const double yMidTop  = yMid + t * 0.5;
  const double yMidBot  = yMid - t * 0.5;
  double xMidR          = xStemR + w * mid_len_rel;
  if (xMidR > xR) xMidR = xR;     // 不超过右边界
  if (xMidR < xStemR + t) xMidR = xStemR + t; // 至少比竖干伸出一点

  // 组合“竖干 + 顶横梁 + 中横梁”的并集外边界，构成单一凹多边形
  // 走向：从左下角逆时针围出整个 F 的外轮廓（无自交）
  PolygonalShape s;
  std::vector<Eigen::Vector2d> P = {
    // 1) 左侧自下而上
    {xL, yB},
    {xL, yT},

    // 2) 顶边右行（顶横梁上缘）
    {xR, yT},

    // 3) 顶横梁下缘左行到竖干右缘
    {xR, yTopBot},
    {xStemR, yTopBot},

    // 4) 沿竖干右缘向下走到“中横梁上缘”高度
    {xStemR, yMidTop},

    // 5) 走中横梁上缘到右端
    {xMidR, yMidTop},

    // 6) 下到中横梁下缘，再回到竖干右缘
    {xMidR, yMidBot},
    {xStemR, yMidBot},

    // 7) 竖干右缘继续下到底边，再回左下角闭合
    {xStemR, yB},
    {xL, yB}
  };

  s.outer_m_ = std::move(P);
  return s;
}

  // 放大 A（可带三角孔）
  static PolygonalShape MakeA(double w, double h, double t, bool with_hole=true)
  {
    const double xL = -w/2, xR = +w/2;
    const double yB = -h/2, yT = +h/2;

    PolygonalShape s;
    // 外轮廓：两腿 + 顶点 + 横梁凹台阶（用于增加 A 的“横梁”外观）
    std::vector<Eigen::Vector2d> P = {
      {xL, yB},
      {xL + t, yB},
      {0.0, yT},
      {xR - t, yB},
      {xR, yB},
      {0.0, yT - t},
    };
    // 横梁位置与厚度（近似即可）
    const double barY = yB + h*0.50;//0.2
    const double halfBar = w*0.30;
    P.insert(P.begin()+2, { -halfBar, barY + t/2 });
    P.insert(P.begin()+3, { +halfBar, barY + t/2 });
    P.insert(P.begin()+4, { +halfBar, barY - t/2 });
    P.insert(P.begin()+5, { -halfBar, barY - t/2 });

    s.outer_m_ = std::move(P);

    if (with_hole) {
      // 中部三角孔
      std::vector<Eigen::Vector2d> hole = {
        { 0.0, yT - t*2.2 },
        { -t*1.6, yT - t*5.2 },
        { +t*1.6, yT - t*5.2 },
      };
      s.holes_m_.push_back(std::move(hole));
    }
    return s;
  }
};
