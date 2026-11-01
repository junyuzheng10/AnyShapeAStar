#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "gridmap.hpp"
#include "shape.hpp"
#include "astar.hpp"

enum class ShapeChoice { FShape, AShape, LShape, VehicleShape, VLineShape };

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

// —— 清理帧目录中的旧帧（上次运行残留帧会混入本次 GIF 结尾）——
static void clearOldFrames(const std::string& path) {
  DIR* dir = ::opendir(path.c_str());
  if (!dir) return;
  while (dirent* ent = ::readdir(dir)) {
    const std::string name = ent->d_name;
    if (name.rfind("frame_", 0) == 0 &&
        name.size() > 4 && name.compare(name.size() - 4, 4, ".png") == 0) {
      ::unlink((path + "/" + name).c_str());
    }
  }
  ::closedir(dir);
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
  const ShapeChoice shape_choice = ShapeChoice::VLineShape; // VLine / Vehicle / L / F / A
  const int W = 800, H = 500;
  const double mpp = 0.05;      // 5 cm / pixel
  const int theta_bins = 36;    // 10°
  const double step_pix = 1.0;  // 单次递进 1px（更精细，求解时间相应变长）

  // GIF 导出参数
  const bool save_gif = true;               // 开启/关闭 GIF 生成
  const int gif_fps   = 30;                 // 帧率
  const int frame_stride = 1;               // 每多少个路径点存一帧（1 = 每点都存）
  const std::string frames_dir = "anim_frames"; // 相对运行目录（通常是 build/）
  const std::string gif_out   = "path_animation.gif";

   // ===== 地图 =====
  GridMap gm(W, H, mpp);
  // 上下两道横墙从左边界伸入，右端折成竖干（┘/┴ 形），中间为水平走廊
  // 墙体厚度 6px（原 12px 减半）；走廊内缘 225/285 与凸块顶 273 保持不变，
  // 因此走廊 60px 间隙、凸块上方 48px 窗口、起点贴墙约束均不受影响
  // 上墙：横笔从左边界到 x=366，右端竖干向上到顶
  gm.addRect(  0, 219, 366,  6);   // 横笔（下缘 225 = 走廊顶，厚 6px）
  gm.addRect(360,   0,   6, 219);  // 竖干
  // 下墙：横笔从左边界到 x=378，竖干向下到底
  gm.addRect(  0, 285, 378,  6);   // 横笔（上缘 285 = 走廊底，厚 6px）
  gm.addRect(360, 291,   6, 209);  // 竖干（291 → 底）
  // 下墙凸出：在竖干位置向上凸起至 y=273（上方仍留 48px 窗口），x 向宽度 6px
  gm.addRect(360, 273,   6, 12);   // 凸出块（273 → 285）
  // 走廊自由带：y ∈ (225, 285)，高 60px = 3.0m；凸块上方收窄为 48px
  gm.computeDistanceField(); // 预计算距离场：用于“远离障碍”软惩罚


  // ===== 机器人形状（放大）=====
  PolygonalShape agent;
  if (shape_choice == ShapeChoice::VLineShape) {
    // 运动物体：竖杆（下段 1.05m、上段 1.5m、厚 0.5m=10px，总高 51px > 48px 窗口
    //          → 直立无法过缝；加厚后需后倾 ≥40° 缩短竖向投影，旋转成为必需动作）
    //          + 右伸横线（长 7.5m）+ 横线上“低-高-低”空心轮廓：
    //   矮矩形 0.9m 高贯通 1.5m→7.5m；中部加高框 1.8m 高（顶贴上墙）位于 4.0m→6.4m
    agent = PolygonalShape::MakeVHLineHump(/*h_below=*/1.05, /*h_above=*/1.50, /*len=*/7.5,
                                           /*mast_w=*/0.5,
                                           /*rect_x0=*/1.5, /*rect_h=*/0.9,
                                           /*box_x0=*/4.0, /*box_x1=*/6.4, /*box_h=*/1.8,
                                           /*t=*/0.1);
  } else if (shape_choice == ShapeChoice::VehicleShape) {
    // 车辆侧影（0.7 倍等比缩小，保证 60px 走廊内几何可解）：
    // 车长 7.7m(154px)、尾鳍 1.89m(37.8px)、车体高 1.4m(28px)、笔画 0.21m
    // 直行过凸块需车身抬升：y∈[264,272] 时车身/尾鳍均可越过 48px 豁口
    agent = PolygonalShape::MakeVehicle(/*L=*/7.7, /*h_fin=*/1.89, /*h_body=*/1.4, /*t=*/0.21);
  } else if (shape_choice == ShapeChoice::LShape) {
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
  planner.w_yaw = 20.0;  // 旋转代价：原地转 10° ≈ 平移 20px，强约束“尽量平移不旋转”
  planner.w_clear    = 50.0;  // 远离障碍软惩罚权重：每行进 1px 每缺 1px 间隙罚 50
                              // （1px 缺口 ≈ 2.5 次旋转）——量级远大于旋转，保证优先离障
  planner.w_smooth   = 0.5;   // 平滑度惩罚：平移↔旋转动作每次切换罚 0.5（轻微抑制频繁切换），
                              // 量级远低于旋转与离障
  planner.w_time     = 10.0;  // 完成时间惩罚：平移 1px 罚 10、旋转一档罚 200（w_yaw=20 等效时间），
                              // “最快完成”主导所有软惩罚；量级低于离障(50/px)
  planner.w_dir      = 2.0;   // x/y 运动惯性：连续平移方向切换罚 2（≈平移 2px），
                              // 鼓励直线平移；量级低于旋转(20)与离障(50/px)
  planner.w_line     = 5.0;   // x/y 趋于终端期望（与 yaw 期望姿态惩罚同构）：每步按偏离
                              // “起点→终点连线”的垂距直接罚 5（1px 偏离=5，2px≈一次旋转），
                              // 强度大于惯性(2/次)、远小于离障(50/px)
  planner.w_pose     = 1.0;   // 非期望姿态惩罚：偏离期望姿态（直立）每 bin 每行进 1px 罚 1，
                              // 驱使物体只在必要时倾斜、且尽快回正（低于旋转量级）
  planner.th_des     = 0;     // 期望姿态：直立（th=0）
  planner.d_safe_pix = 10.0;  // 期望安全间隙 10px = 0.5m
  // yaw 边界：只允许 [-40°, +40°]（bin 32..35 = -40°..-10°，0..4 = 0..+40°），防止无限旋转
  planner.th_lo = theta_bins - 4;   // 32
  planner.th_hi = 4;
  planner.visualize_expand = false; // 只显示最终结果

  // ===== 起终点 =====
  // 起点：竖线下端贴合下墙（下墙顶缘 y=285）。线高 42px，中心 y=263 时
  // 线段占据 y∈[242,284]，最下沿像素紧邻墙面（间隙 1px，视觉上接触；
  // 规划器将间隙=0 判为碰撞，故取 1px）
  Eigen::Vector3i start(60, 263, 0);
  Eigen::Vector3i goal (560, 255, 0); // 驶出豁口后的停车点（不得超出地图）

  // ===== 规划（仅统计纯规划耗时，不含可视化/图片生成）=====
  const auto t0 = std::chrono::steady_clock::now();
  auto path = planner.plan(start, goal);
  const auto t1 = std::chrono::steady_clock::now();
  const double plan_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "[time] planning: " << plan_ms << " ms" << std::endl;

  // ===== 可视化（最终路径 & 动画帧生成）=====
  cv::Mat base; // 仅背景
  cv::cvtColor(gm.occ, base, cv::COLOR_GRAY2BGR);
  cv::circle(base, {start.x(),start.y()}, 4, {0,255,0}, cv::FILLED);
  cv::circle(base, {goal.x(),goal.y()},   4, {0,0,255}, cv::FILLED);

  // —— 初始状态 PNG（不论是否求解成功均输出）——
  {
    cv::Mat init_img = base.clone();
    const double yaw_start = (2.0 * M_PI / planner.theta_bins) * start.z();
    drawAgentAt(init_img, agent, yaw_start, start.x(), start.y(), gm.meter_per_pixel);
    cv::imwrite("start_state.png", init_img);
    std::cout << "[png] 已保存初始状态: start_state.png" << std::endl;
  }

  if (path.empty()) {
    std::cout << "A* failed: no path found.\n";
    // —— 结束状态 PNG（无解时同样输出，画出目标姿态）——
    cv::Mat goal_img = base.clone();
    const double yaw_goal = (2.0 * M_PI / planner.theta_bins) * goal.z();
    drawAgentAt(goal_img, agent, yaw_goal, goal.x(), goal.y(), gm.meter_per_pixel);
    cv::imwrite("final_state.png", goal_img);
    std::cout << "[png] 已保存结束状态（无路径）: final_state.png" << std::endl;
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

  // —— 结束状态 PNG（求解成功：含完整路径与终点姿态）——
  cv::imwrite("final_state.png", final_img);
  std::cout << "[png] 已保存结束状态: final_state.png" << std::endl;

  // —— 生成动画帧（沿路径逐步增长）——
  if (save_gif) {
    clearOldFrames(frames_dir); // 先清掉上次运行残留的旧帧，避免 GIF 结尾混入旧画面
    if (!ensureDir(frames_dir)) {
      std::cerr << "[gif] 无法创建目录：" << frames_dir << std::endl;
    } else {
      int frame_idx = 0;
      cv::Mat frame;
      const double rot_cap_deg = 1.0;  // 每帧最大旋转角（度）：动画转速上限，调小则旋转更慢
      double prev_yaw = (2.0 * M_PI / planner.theta_bins) * path[0].z();

      for (size_t k=1; k<path.size(); ++k) {
        if (k % frame_stride != 0 && k+1 < path.size()) continue;

        // 旋转步骤按 rot_cap_deg/帧 插值拆帧，限制动画最高角速度（值越小旋转越慢）；
        // 平移步骤保持每节点 1 帧
        const double yaw = (2.0 * M_PI / planner.theta_bins) * path[k].z();
        double dyaw = std::atan2(std::sin(yaw - prev_yaw), std::cos(yaw - prev_yaw));
        int rot_sub = 1;
        if (std::abs(dyaw) > 1e-9) {
          rot_sub = std::max(1, (int)std::ceil(std::abs(dyaw) * 180.0 / M_PI / rot_cap_deg));
        }

        for (int sub = 1; sub <= rot_sub; ++sub) {
          frame = base.clone();

          // 画从起点到第 k 段的路径
          for (size_t i=1; i<=k; ++i) {
            cv::line(frame,
                     {path[i-1].x(), path[i-1].y()},
                     {path[i].x(),   path[i].y()},
                     {0,0,255}, 2);
          }

          // 当前姿态下画机器人形状（描边）；旋转帧取插值角度
          const double yaw_i = (rot_sub == 1) ? yaw : prev_yaw + dyaw * sub / rot_sub;
          drawAgentAt(frame, agent, yaw_i, path[k].x(), path[k].y(),
                      gm.meter_per_pixel, /*color=*/{50,180,255}, /*thick=*/2);

          // 保存 PNG 帧（零填充编号以保证字典序）
          char namebuf[256];
          std::snprintf(namebuf, sizeof(namebuf), "%s/frame_%05d.png", frames_dir.c_str(), frame_idx++);
          cv::imwrite(namebuf, frame);
        }
        prev_yaw = yaw;
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
