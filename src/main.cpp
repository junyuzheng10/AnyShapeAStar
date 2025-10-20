#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "gridmap.hpp"
#include "shape.hpp"
#include "astar.hpp"

enum class ShapeChoice { FShape, AShape };

// —— 在图像上画出机器人形状（含内孔 / 外轮廓描边）——
static void drawAgentAt(cv::Mat& vis,
                        const PolygonalShape& agent,
                        double yaw_rad,
                        int cx, int cy,
                        double meter_per_pixel,
                        const cv::Scalar& color = {50,180,255},
                        int thickness = 2)
{
  // 外轮廓
  auto outer = PolygonalShape::rotateToPixels(agent.outer_m_, yaw_rad, meter_per_pixel);
  std::vector<cv::Point> outer_img; outer_img.reserve(outer.size());
  for (auto&p: outer) outer_img.emplace_back(p.x + cx, p.y + cy);
  const std::vector<std::vector<cv::Point>> polys{outer_img};
  cv::polylines(vis, polys, true, color, thickness);

  // 内孔
  for (auto& hole_m : agent.holes_m_) {
    auto hole = PolygonalShape::rotateToPixels(hole_m, yaw_rad, meter_per_pixel);
    std::vector<cv::Point> hole_img; hole_img.reserve(hole.size());
    for (auto&p: hole) hole_img.emplace_back(p.x + cx, p.y + cy);
    const std::vector<std::vector<cv::Point>> holes{hole_img};
    cv::polylines(vis, holes, true, cv::Scalar(200,200,50), 1);
  }
}

// —— 检查命令是否可用（用于选择 convert/ffmpeg）——
static bool commandExists(const std::string& cmd) {
  std::string check = "which " + cmd + " > /dev/null 2>&1";
  int ret = std::system(check.c_str());
  return (ret == 0);
}

// —— 创建目录（无 <filesystem>，使用 POSIX mkdir）——
static bool ensureDir(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0) {
    return (st.st_mode & S_IFDIR) != 0;
  }
  // 0755 权限；若需要 “-p” 效果可改为 system("mkdir -p ...")
  return ::mkdir(path.c_str(), 0755) == 0;
}

// —— 从 PNG 合成为 GIF（优先 ffmpeg，次选 convert）——
static bool makeGifFromPngs(const std::string& frames_dir,
                            const std::string& out_gif,
                            int fps = 15)
{
  if (commandExists("ffmpeg")) {
    std::string cmd =
      "ffmpeg -y -framerate " + std::to_string(fps) +
      " -i " + frames_dir + "/frame_%05d.png "
      "-filter_complex \"fps=" + std::to_string(fps) +
      ",split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" " + out_gif;
    std::cout << "[gif] run: " << cmd << std::endl;
    return (std::system(cmd.c_str()) == 0);
  }
  if (commandExists("convert")) {
    int delay_cs = 100 / fps; // 1/100 秒为单位的帧延时
    std::string cmd =
      "convert -delay " + std::to_string(delay_cs) +
      " -loop 0 " + frames_dir + "/frame_*.png " + out_gif;
    std::cout << "[gif] run: " << cmd << std::endl;
    return (std::system(cmd.c_str()) == 0);
  }
  std::cerr << "[gif] 未找到 ffmpeg 或 convert，已保存 PNG 帧在目录：" << frames_dir << std::endl;
  return false;
}

int main() {
  // ===== 参数设置 =====
  const ShapeChoice shape_choice = ShapeChoice::FShape; // 改成 FShape 可切换“大 F”
  const int W = 800, H = 500;
  const double mpp = 0.05;      // 5 cm / pixel
  const int theta_bins = 36;    // 10°
  const double step_pix = 5.0;  // 更大机器人 -> 更大步长

  // GIF 导出参数
  const bool save_gif = true;               // 开启/关闭 GIF 生成
  const int gif_fps   = 30;                 // 帧率
  const int frame_stride = 1;               // 每多少个路径点存一帧（1 = 每点都存）
  const std::string frames_dir = "anim_frames"; // 相对运行目录（通常是 build/）
  const std::string gif_out   = "path_animation.gif";

   // ===== 地图 =====
  GridMap gm(W, H, mpp);
  //gm.addRandomBlocks(16, 20, 14, 110, 70, /*seed*/7);
  //gm.addRandomCircles(12, 9, 26, /*seed*/11);
 // gm.addRandomDots(100, /*seed*/29);   // 零星小点
  gm.addRandomCircles(70, 5, 10, /*seed*/10);


  // ===== 机器人形状（放大）=====
  PolygonalShape agent;
  if (shape_choice == ShapeChoice::FShape) {
    // F：约 1.8m x 2.2m，笔画厚度 0.34m
//agent = PolygonalShape::MakeF(/*w=*/1.80, /*h=*/2.20, /*t=*/0.34);
// 或者微调中横梁位置和长度（比如更靠下、更长）：
    agent = PolygonalShape::MakeF(1.80, 2.20, 0.34, /*mid_rel=*/0.45, /*mid_len_rel=*/0.70);
  } else {
    // A：约 2.0m x 2.4m，笔画厚度 0.32m（带孔）
    agent = PolygonalShape::MakeA(/*w=*/2.00, /*h=*/3.40, /*t=*/0.32, /*with_hole=*/true);
  }

  // ===== 规划器 =====
  AnyShapeAStar planner(gm, agent);
  planner.theta_bins = theta_bins;
  planner.step_pix   = step_pix;
  planner.yaw_step_bins = 1;
  planner.w_yaw = 0.010;
  planner.visualize_expand = false; // 只显示最终结果

  // ===== 起终点 =====
  Eigen::Vector3i start(40, H-40, 0);
  Eigen::Vector3i goal (W-60, 60, theta_bins/4); // 朝上（~90°）

  // ===== 规划 =====
  auto path = planner.plan(start, goal);

  // ===== 可视化（最终路径 & 动画帧生成）=====
  cv::Mat base; // 仅背景
  cv::cvtColor(gm.occ, base, cv::COLOR_GRAY2BGR);
  cv::circle(base, {start.x(),start.y()}, 4, {0,255,0}, cv::FILLED);
  cv::circle(base, {goal.x(),goal.y()},   4, {0,0,255}, cv::FILLED);

  if (path.empty()) {
    std::cout << "A* failed: no path found.\n";
    cv::imshow("AnyShape A*", base);
    cv::waitKey(0);
    return 0;
  }

  // —— 最终结果图（静态展示）——
  cv::Mat final_img = base.clone();
  for (size_t i=1;i<path.size();++i) {
    cv::line(final_img,
             {path[i-1].x(), path[i-1].y()},
             {path[i].x(),   path[i].y()},
             {0,0,255}, 2);
  }
  // 终点姿态画形状
  {
    const double yaw_goal = (2.0 * M_PI / planner.theta_bins) * goal.z();
    drawAgentAt(final_img, agent, yaw_goal, goal.x(), goal.y(), gm.meter_per_pixel, {0,140,255}, 2);
  }

  // —— 生成动画帧（沿路径逐步增长）——
  if (save_gif) {
    if (!ensureDir(frames_dir)) {
      std::cerr << "[gif] 无法创建目录：" << frames_dir << std::endl;
    } else {
      int frame_idx = 0;
      cv::Mat frame;

      for (size_t k=1; k<path.size(); ++k) {
        if (k % frame_stride != 0 && k+1 < path.size()) continue;

        frame = base.clone();

        // 画从起点到第 k 段的路径
        for (size_t i=1; i<=k; ++i) {
          cv::line(frame,
                   {path[i-1].x(), path[i-1].y()},
                   {path[i].x(),   path[i].y()},
                   {0,0,255}, 2);
        }

        // 当前姿态下画机器人形状（描边）
        const int cx = path[k].x();
        const int cy = path[k].y();
        const int th = path[k].z();
        const double yaw = (2.0 * M_PI / planner.theta_bins) * th;
        drawAgentAt(frame, agent, yaw, cx, cy, gm.meter_per_pixel, /*color=*/{50,180,255}, /*thick=*/2);

        // 保存 PNG 帧（零填充编号以保证字典序）
        char namebuf[256];
        std::snprintf(namebuf, sizeof(namebuf), "%s/frame_%05d.png", frames_dir.c_str(), frame_idx++);
        cv::imwrite(namebuf, frame);
      }

      // 合成为 GIF
      if (makeGifFromPngs(frames_dir, gif_out, gif_fps)) {
        std::cout << "[gif] 已生成 GIF: " << gif_out << std::endl;
      } else {
        std::cout << "[gif] 未自动生成 GIF。你可手动执行（任选其一）:\n"
                     "  ffmpeg -y -framerate " << gif_fps
                  << " -i " << frames_dir << "/frame_%05d.png "
                     "-filter_complex \"fps=" << gif_fps
                  << ",split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" "
                  << gif_out << "\n"
                     "或：\n"
                     "  convert -delay " << (100/gif_fps)
                  << " -loop 0 " << frames_dir << "/frame_*.png " << gif_out << "\n";
      }
    }
  }

  // —— 展示静态最终图 —— 
  cv::imshow("AnyShape A*", final_img);
  std::cout << "Path length (nodes): " << path.size() << std::endl;
  cv::waitKey(0);
  return 0;
}
