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
   * 间隙查询：把形状以 (cx,cy) 为形心、yaw_rad 朝向放置到地图上，
   * 返回足迹内像素到最近障碍的最小距离（像素，基于预计算的 dist 距离场）。
   * @param dist  距离场（GridMap::computeDistanceField 生成）
   * @return 最小间隙；0 表示碰撞（足迹压到障碍或越界）
   */
  double clearanceAt(const cv::Mat& map_bin, const cv::Mat& dist,
                     int cx, int cy,
                     double yaw_rad, double meter_per_pixel) const
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

    // 越界视作碰撞（间隙为 0）
    if (roi.x < 0 || roi.y < 0 ||
        roi.x + roi.width > map_bin.cols ||
        roi.y + roi.height > map_bin.rows)
      return 0.0;

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

    // 4) 足迹内距离场最小值 = 车体上离障碍最近的点
    double minD = 0.0;
    cv::minMaxIdx(dist(roi), &minD, nullptr, nullptr, nullptr, shapeMask);
    return minD;
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

  // 车辆侧影（俯视平面形状）：尾部高竖鳍 + 底部长条 + 前部甲板 + 甲板上的凸块
  // 单位：米。局部原点在尾鳍根部（后下角），x 向前，y 向下（图像约定，上方为 -y）
  // L: 总长; h_fin: 尾鳍总高（≈走廊宽，直立刚好能开）; h_body: 车体总高（≈凸块上方间隙）; t: 笔画厚
  static PolygonalShape MakeVehicle(double L, double h_fin, double h_body, double t)
  {
    const double x_deck = L * 0.40;   // 甲板起点（前部 60% 抬高）
    const double x_b0   = L * 0.68;   // 凸块后缘
    const double x_b1   = L * 0.80;   // 凸块前缘
    const double h_deck = h_body - t; // 甲板高（凸块再高出 t）

    PolygonalShape s;
    std::vector<Eigen::Vector2d> P = {
      {0, 0},            // 后下角（鳍根）
      {L, 0},            // 车头底角
      {L, -h_deck},      // 车头立面到甲板顶
      {x_b1, -h_deck},   // 甲板顶左行到凸块右缘
      {x_b1, -h_body},   // 凸块右缘上行
      {x_b0, -h_body},   // 凸块顶左行
      {x_b0, -h_deck},   // 凸块左缘下行
      {x_deck, -h_deck}, // 甲板顶左行到甲板左缘
      {x_deck, -t},      // 甲板左缘下行到底条顶
      {t, -t},           // 底条顶左行到鳍右缘
      {t, -h_fin},       // 鳍右缘上行
      {0, -h_fin},       // 鳍顶
    };
    s.outer_m_ = std::move(P);
    return s;
  }

  // 竖直线段（最简运动物体）：一条高 h、厚 t 的细竖条，中心在原点
  // h≈走廊宽（两边间距），直立时上下各留少量间隙
  static PolygonalShape MakeVLine(double h, double t)
  {
    PolygonalShape s;
    std::vector<Eigen::Vector2d> P = {
      {0, -h/2}, {t, -h/2}, {t, h/2}, {0, h/2},
    };
    s.outer_m_ = std::move(P);
    return s;
  }

  // 竖线 + 从竖线中心向右伸出的横线（“⊢”形）：竖线高 h，横线长 len，
  // 笔画厚 t，整体中心在原点（竖线中心与横线左端同高）
  static PolygonalShape MakeVHLine(double h, double len, double t)
  {
    PolygonalShape s;
    std::vector<Eigen::Vector2d> P = {
      {0, -h/2}, {t, -h/2}, {t, -t/2}, {len, -t/2},
      {len,  t/2}, {t,  t/2}, {t,  h/2}, {0,  h/2},
    };
    s.outer_m_ = std::move(P);
    return s;
  }

  // 竖线 + 右伸横线 + 横线右端上方的方形框（空心，壁厚 t）：
  // 方框底边贴横线上缘。框高受两处约束：走廊巡航（框顶≥上墙下缘+1 且竖线底≤下墙上缘-1）
  // 与窄缝窗口（框+横线总高 ≤ 48px）
  static PolygonalShape MakeVHLineBox(double h, double len, double box_w, double box_h, double t)
  {
    PolygonalShape s;
    const double yTop = -t/2 - box_h;   // 方框顶（横线上方，y 向下为正）
    std::vector<Eigen::Vector2d> P = {
      {0,  h/2}, {t,  h/2}, {t,  t/2}, {len,  t/2},
      {len, yTop}, {len - box_w, yTop},
      {len - box_w, -t/2}, {t, -t/2}, {t, -h/2}, {0, -h/2},
    };
    s.outer_m_ = std::move(P);
    // 方框内孔（四壁厚 t）
    std::vector<Eigen::Vector2d> hole = {
      {len - box_w + t, yTop + t}, {len - t, yTop + t},
      {len - t, -t/2 - t}, {len - box_w + t, -t/2 - t},
    };
    s.holes_m_.push_back(std::move(hole));
    return s;
  }

  // 竖线 + 右伸横线 + 横线上两个骑跨框（均空心、壁厚 t）：
  //   大方框 box_w×box_h 紧贴横线右端上方；
  //   矮矩形框 rect_w×rect_h 位于其左侧（左缘 rect_x0），底边同贴横线上缘
  static PolygonalShape MakeVHLineTwoBox(double h, double len,
                                         double box_w, double box_h,
                                         double rect_x0, double rect_w, double rect_h,
                                         double t)
  {
    PolygonalShape s;
    const double boxTop  = -t/2 - box_h;    // 大方框顶
    const double rectTop = -t/2 - rect_h;   // 矮框顶
    const double rectX1  = len - box_w;     // 矮框右缘（与大框左缘齐平）
    std::vector<Eigen::Vector2d> P = {
      {0, h/2}, {t, h/2}, {t, t/2}, {len, t/2},
      {len, boxTop}, {len - box_w, boxTop},
      {rectX1, rectTop}, {rect_x0, rectTop},
      {rect_x0, -t/2}, {t, -t/2}, {t, -h/2}, {0, -h/2},
    };
    s.outer_m_ = std::move(P);
    // 大方框内孔
    s.holes_m_.push_back({
      {len - box_w + t, boxTop + t}, {len - t, boxTop + t},
      {len - t, -t/2 - t}, {len - box_w + t, -t/2 - t},
    });
    // 矮框内孔
    s.holes_m_.push_back({
      {rect_x0 + t, rectTop + t}, {rectX1 - t, rectTop + t},
      {rectX1 - t, -t/2 - t}, {rect_x0 + t, -t/2 - t},
    });
    return s;
  }

  // 竖杆（下段 h_below、上段 h_above，宽 mast_w 可不对称加宽）+ 右伸横线
  // + 横线上矮矩形（贯通到右端）+ 中部加高框（“低-高-低”轮廓）：均空心、壁厚 t
  // 注：杆加宽后倾斜过缝的竖向投影 = 杆长·cosθ + 杆宽·sinθ，倾角需求随宽度增大
  static PolygonalShape MakeVHLineHump(double h_below, double h_above, double len,
                                       double mast_w,
                                       double rect_x0, double rect_h,
                                       double box_x0, double box_x1, double box_h,
                                       double t)
  {
    PolygonalShape s;
    const double rectTop = -t/2 - rect_h;   // 矮矩形顶
    const double boxTop  = -t/2 - box_h;    // 加高框顶
    std::vector<Eigen::Vector2d> P = {
      {0, h_below}, {mast_w, h_below}, {mast_w, t/2}, {len, t/2},
      {len, rectTop}, {box_x1, rectTop}, {box_x1, boxTop},
      {box_x0, boxTop}, {box_x0, rectTop}, {rect_x0, rectTop},
      {rect_x0, -t/2}, {mast_w, -t/2}, {mast_w, -h_above}, {0, -h_above},
    };
    s.outer_m_ = std::move(P);
    // 矮矩形内孔（被加高框分成左右两段）
    s.holes_m_.push_back({
      {rect_x0 + t, rectTop + t}, {box_x0 - t, rectTop + t},
      {box_x0 - t, -t/2 - t}, {rect_x0 + t, -t/2 - t},
    });
    s.holes_m_.push_back({
      {box_x1 + t, rectTop + t}, {len - t, rectTop + t},
      {len - t, -t/2 - t}, {box_x1 + t, -t/2 - t},
    });
    // 加高框内孔
    s.holes_m_.push_back({
      {box_x0 + t, boxTop + t}, {box_x1 - t, boxTop + t},
      {box_x1 - t, -t/2 - t}, {box_x0 + t, -t/2 - t},
    });
    return s;
  }

  // L 形（竖干在左、底横笔向右）：宽 w、高 h、笔画厚度 t
  static PolygonalShape MakeL(double w, double h, double t)
  {
    const double xL = -w/2, xR = +w/2;
    const double yB = -h/2, yT = +h/2;

    PolygonalShape s;
    // 注意：像素/图像坐标系 y 轴向下，“底部横笔”须放在 +y（yT）一侧
    // 外轮廓：左缘上行 -> 竖干顶边右行 -> 竖干内缘下行 -> 横笔上缘右行 -> 横笔右缘下行 -> 底边左行闭合
    std::vector<Eigen::Vector2d> P = {
      {xL, yT},
      {xL, yB},
      {xL + t, yB},
      {xL + t, yT - t},
      {xR, yT - t},
      {xR, yT},
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
