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
 * - 邻接动作（世界系控制，与朝向解耦）：±x 平移、±y 平移、原地 yaw 旋转
 * - 碰撞检测：调用 PolygonalShape::collidesAt
 * - 只输出最终路径（不显示展开过程）
 */

struct Node {
  double g{std::numeric_limits<double>::infinity()};
  double g_base{std::numeric_limits<double>::infinity()}; // 不含离障惩罚的基础代价（去重用）
  double f{std::numeric_limits<double>::infinity()};
  int parent{-1};   // 状态索引（-1 = 无父节点）；坐标/姿态由索引解码
  signed char dir{0}; // 上次平移方向：0 无 / 1 +x / 2 -x / 3 +y / 4 -y（旋转动作继承）
  bool closed{false};
  bool opened{false};
};

struct AnyShapeAStar {
  const GridMap& gm;
  const PolygonalShape& shape;
  int    theta_bins{36};
  double step_pix{5.0};      // 机器人更大 -> 步长略大
  double yaw_step_bins{1};
  double w_clear{0.5};        // 远离障碍软惩罚权重（间隙每缺 1px 罚 w_clear）
  double w_dir{0.0};          // x/y 运动惯性：连续平移方向切换、原地旋转每档各罚 w_dir
                              // （旋转不打断平移惯性，方向继承父节点；量级须低于离障）
  double d_safe_pix{10.0};    // 期望安全间隙（像素），间隙低于此值才产生惩罚
  int    th_lo{0};            // 允许的最小 yaw bin（含）
  int    th_hi{35};           // 允许的最大 yaw bin（含）；th_lo > th_hi 表示窗口跨 0 环绕
  bool   visualize_expand{false}; // 本工程仅最终路径显示

  AnyShapeAStar(const GridMap& map, const PolygonalShape& sh)
    : gm(map), shape(sh) {}

  inline double yawOf(int th) const {
    const double d = (2.0*M_PI)/theta_bins;
    return d * th;
  }

  // yaw 边界检查：窗口 [th_lo, th_hi]，可跨 0 环绕（th_lo > th_hi 时）
  inline bool thAllowed(int th) const {
    if (th_lo <= th_hi) return th >= th_lo && th <= th_hi;
    return th >= th_lo || th <= th_hi;
  }

  struct PQItem { double f; int idx; bool operator<(const PQItem& o) const { return f > o.f; } };

  // ===== 状态索引（紧凑编码）=====
  // 只为 yaw 窗口 [th_lo, th_hi]（可跨 0 环绕）内的 bin 分配状态，
  // 内存从 W*H*theta_bins 降为 W*H*nRanks（36 档窗口 9 档时约 1/4）。
  // 坐标/姿态均可从状态索引直接解码，无需在 Node 中冗余存储。
  inline int nRanks() const {
    return (th_lo <= th_hi) ? (th_hi - th_lo + 1)
                            : (theta_bins - th_lo + th_hi + 1);
  }
  inline int rankOf(int th) const {
    return (th_lo <= th_hi) ? (th - th_lo)
                            : (th >= th_lo ? th - th_lo
                                           : th + (theta_bins - th_lo));
  }
  inline int thOfRank(int r) const { return (th_lo + r) % theta_bins; }

  inline int idx(int x,int y,int th) const { return (rankOf(th)*gm.H + y)*gm.W + x; }
  inline void decode(int id, int& x, int& y, int& th) const {
    x = id % gm.W;
    y = (id / gm.W) % gm.H;
    th = thOfRank(id / (gm.W * gm.H));
  }

  // ===== C-space 预计算（加速碰撞查询）=====
  // 每档姿态只做一次 fillPoly 预烘焙"实心足迹掩膜"（含外轮廓、扣除内孔），
  // 之后 clearance() 查询 = 按掩膜行段扫描距离场取最小值：
  //   - 碰撞（dist=0）提前退出，大多数被拒绝的状态只需扫几个像素
  //   - 行段连续内存，顺序读取对缓存/SIMD 友好
  // 数值上与原 fillPoly+minMaxIdx 逐位一致（同一掩膜构造），不改变搜索结果
  struct BinFootprint {
    cv::Mat mask;        // CV_8UC1 实心足迹（bbox 局部坐标）
    std::vector<cv::Point> contour; // 外层连通轮廓离散点云（bbox 局部坐标，不含孔洞边界）
    cv::Point org;       // 形状中心（局部原点）在掩膜内的位置
    bool ready{false};
  };

  // 轮廓点云抽稀：贪心保留，删除与已保留点邻近（切比雪夫距离 ≤ min_gap）的像素
  static std::vector<cv::Point> thinContour(const std::vector<cv::Point>& pts,
                                            int min_gap = 1) {
    std::vector<cv::Point> out;
    out.reserve(pts.size() / 2 + 1);
    for (const auto& p : pts) {
      bool near = false;
      for (const auto& q : out) {
        if (std::max(std::abs(p.x - q.x), std::abs(p.y - q.y)) <= min_gap) {
          near = true;
          break;
        }
      }
      if (!near) out.push_back(p);
    }
    return out;
  }

  std::vector<BinFootprint> fp_;
  // 性能计数（供基准测试观察，不影响结果）
  long long n_clr_calls{0}, n_clr_pix{0};

  void buildFootprint(int th) {
    if ((int)fp_.size() < theta_bins) fp_.resize(theta_bins);
    BinFootprint& f = fp_[th];
    if (f.ready) return;
    const double yaw = yawOf(th);
    auto outer = PolygonalShape::rotateToPixels(shape.outer_m_, yaw, gm.meter_per_pixel);
    std::vector<std::vector<cv::Point>> holes;
    holes.reserve(shape.holes_m_.size());
    for (auto& h : shape.holes_m_)
      holes.push_back(PolygonalShape::rotateToPixels(h, yaw, gm.meter_per_pixel));
    cv::Rect bbox = cv::boundingRect(outer);
    for (auto& hp : holes) bbox |= cv::boundingRect(hp);
    // 平移到 bbox 局部坐标（与原 clearanceAt 的 shiftToROI 一致）
    for (auto& p : outer) { p.x -= bbox.x; p.y -= bbox.y; }
    for (auto& hp : holes)
      for (auto& p : hp) { p.x -= bbox.x; p.y -= bbox.y; }
    cv::Mat m = cv::Mat::zeros(bbox.height, bbox.width, CV_8UC1);
    const std::vector<std::vector<cv::Point>> outv{outer};
    cv::fillPoly(m, outv, cv::Scalar(255));
    for (auto& hp : holes) {
      const std::vector<std::vector<cv::Point>> holev{hp};
      cv::fillPoly(m, holev, cv::Scalar(0));
    }
    f.mask = m;
    // 外层连通轮廓的全部像素（RETR_EXTERNAL 不含孔洞边界，CHAIN_APPROX_NONE
    // 保留每个边界像素）：离障软惩罚的边缘点云
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(m.clone(), cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    f.contour.clear();
    for (auto& c : cs) f.contour.insert(f.contour.end(), c.begin(), c.end());
    f.contour = thinContour(f.contour); // 删除邻近像素，得到离散点云
    f.org  = cv::Point(-bbox.x, -bbox.y);
    f.ready = true;
  }

  inline double clearance(int x, int y, int th) {
    ++n_clr_calls;
    buildFootprint(th);
    const BinFootprint& f = fp_[th];
    // 足迹包围盒越界视作碰撞（与原 clearanceAt 语义一致）
    const int x0 = x - f.org.x, y0 = y - f.org.y;
    if (x0 < 0 || y0 < 0 ||
        x0 + f.mask.cols > gm.W || y0 + f.mask.rows > gm.H)
      return 0.0;
    double minD = std::numeric_limits<double>::infinity();
    for (int j = 0; j < f.mask.rows; ++j) {
      const uchar* mrow = f.mask.ptr<uchar>(j);
      const float* drow = gm.dist.ptr<float>(y0 + j);
      int i = 0;
      while (i < f.mask.cols) {
        while (i < f.mask.cols && mrow[i] == 0) ++i;      // 跳过空白段
        if (i >= f.mask.cols) break;
        const int i0 = i;
        while (i < f.mask.cols && mrow[i] != 0) ++i;      // 实心段
        for (int k = i0; k < i; ++k) {
          const float d = drow[x0 + k];
          ++n_clr_pix;
          if (d <= 0.0f) return 0.0;                      // 碰撞：提前退出
          if (d < minD) minD = d;
        }
      }
    }
    return minD;
  }

  inline double heuristic(int x,int y,int gx,int gy) const {
    double dx=x-gx, dy=y-gy;
    // 纯几何引导项：仅欧氏距离（角度项已去除）
    return std::sqrt(dx*dx+dy*dy);
  }

  // ===== line-of-sight 剪枝（仿 ik_test astar_cart_interp_demo.shortcut_path）=====
  // 贪心保留最远可视点，消除 A* 阶梯锯齿：段内 (x,y,th) 线性插值
  // （th 走最短环绕，平移/旋转复合运动），逐像素步进查 clearance，
  // 余量 <= margin_pix 视为不可视。端点由规划保证无碰不复查；
  // 相邻状态剪枝失败时原样保留（退化 = 原序列，不会更差）。
  static int pyRound(double v) {  // Python round 语义：.5 取偶
    const double f = std::floor(v);
    const double d = v - f;
    if (d > 0.5) return (int)f + 1;
    if (d < 0.5) return (int)f;
    return ((int)f % 2 == 0) ? (int)f : (int)f + 1;
  }

  std::vector<Eigen::Vector3i> shortcutPath(
      const std::vector<Eigen::Vector3i>& path, double margin_pix = 1.0)
  {
    if (path.size() <= 2) return path;
    const int bins = theta_bins;

    auto segOk = [&](const Eigen::Vector3i& a, const Eigen::Vector3i& b) {
      const int dth = (((b.z() - a.z() + bins / 2) % bins) + bins) % bins
                      - bins / 2;
      const int n = (int)std::max(
          std::hypot(double(b.x() - a.x()), double(b.y() - a.y())),
          (double)std::max(std::abs(dth), 1));
      for (int k = 1; k < n; ++k) {
        const int u = pyRound(a.x() + (b.x() - a.x()) * k / (double)n);
        const int v = pyRound(a.y() + (b.y() - a.y()) * k / (double)n);
        double thm = std::fmod(a.z() + dth * k / (double)n, bins);
        if (thm < 0) thm += bins;
        const int th = pyRound(thm) % bins;
        if (clearance(u, v, th) <= margin_pix) return false;
      }
      return true;
    };

    std::vector<Eigen::Vector3i> out{path.front()};
    size_t i = 0;
    while (i + 1 < path.size()) {
      size_t j = path.size() - 1;
      while (j > i + 1 && !segOk(path[i], path[j])) --j;
      out.push_back(path[j]);
      i = j;
    }
    return out;
  }

  std::vector<Eigen::Vector3i> plan(
      Eigen::Vector3i start,
      Eigen::Vector3i goal)
  {
    // 起点校验：姿态须在 yaw 窗口内（否则 rankOf 越界），且碰撞（间隙为 0）直接失败
    if (!thAllowed(start.z()) ||
        clearance(start.x(), start.y(), start.z()) <= 0.0)
      return {};

    const int N = gm.W * gm.H * nRanks();
    std::vector<Node> nodes(N);

    auto recPath = [&](int id){
      std::vector<Eigen::Vector3i> path;
      while (id != -1) {
        int x, y, th; decode(id, x, y, th);
        path.emplace_back(x, y, th);
        id = nodes[id].parent;
      }
      std::reverse(path.begin(), path.end());
      return path;
    };

    // pidx: 父节点状态索引（-1 表示起点）；px,py: 父节点坐标
    auto pushOrRelax = [&](int nx,int ny,int nth,double ng_base,
                           int pidx, int px, int py,
                           std::priority_queue<PQItem>& pq){
      if(!gm.inBounds(nx,ny) || !thAllowed(nth)) return;  // 地图边界 + yaw 边界剪枝
      int id = idx(nx,ny,nth);
      if (nodes[id].closed) return;

      // x/y 运动惯性：连续平移方向切换罚 w_dir；原地旋转每档也罚 w_dir
      // （旋转不打断平移惯性，方向继承父节点；量级低于离障/旋转时间代价）
      signed char new_dir = 0;
      if (nx != px || ny != py) {  // 本动作为平移
        new_dir = (nx > px) ? 1 : (nx < px) ? 2 : (ny > py) ? 3 : 4;
        if (w_dir > 0.0 && pidx >= 0) {
          const signed char pdir = nodes[pidx].dir;
          if (pdir > 0 && pdir != new_dir) ng_base += w_dir;
        }
      } else if (w_dir > 0.0 && pidx >= 0) {  // 本动作为原地旋转
        ng_base += w_dir;
      }

      // 去重：离障惩罚只依赖状态 (nx,ny,nth) 本身，故 g = g_base + penalty。
      // 同一状态被多个父节点 push 时，若基础代价已不可能更优，可安全跳过碰撞评估
      if (nodes[id].opened && ng_base >= nodes[id].g_base) return;

      // 足迹到最近障碍的最小间隙：<=0 碰撞（整体实心掩膜判定，含孔洞与越界）
      const double clr = clearance(nx,ny,nth);
      if (clr <= 0.0) return;
      // 软惩罚：外层连通轮廓的离散点云逐点查距离场，各点独立计算
      // max(0, d_safe - d_i) 后求和（贴障的边缘越多罚越重）
      // （clr >= d_safe 时所有点间隙均 >= d_safe，惩罚必为 0，跳过点云遍历）
      double pen = 0.0;
      if (clr < d_safe_pix) {
        const BinFootprint& f = fp_[nth]; // clearance() 已构建该档足迹
        const int px0 = nx - f.org.x, py0 = ny - f.org.y;
        for (const auto& p : f.contour) {
          const float d = gm.dist.ptr<float>(py0 + p.y)[px0 + p.x];
          if (d < d_safe_pix) pen += d_safe_pix - (double)d;
        }
      }
      // 惩罚按行进距离累计（乘 step_pix），保证总代价与步长无关，
      // 否则步长变小时节点数增多会放大惩罚、扭曲最优策略
      const double ng = ng_base + w_clear * step_pix * pen;

      double nf = ng + heuristic(nx,ny,goal.x(),goal.y());
      if (!nodes[id].opened || ng < nodes[id].g) {
        nodes[id].opened = true;
        nodes[id].g = ng; nodes[id].g_base = ng_base; nodes[id].f = nf;
        nodes[id].parent = pidx;
        // 平移动作记录新方向；旋转动作继承父方向
        nodes[id].dir = new_dir ? new_dir
                                : (pidx >= 0 ? nodes[pidx].dir : (signed char)0);
        pq.push({nf, id});
      }
    };

    // 初始化
    std::priority_queue<PQItem> pq;
    const int sid = idx(start.x(),start.y(),start.z());
    nodes[sid].opened=true;
    nodes[sid].g=0.0;
    nodes[sid].g_base=0.0;
    nodes[sid].parent=-1;
    nodes[sid].f=heuristic(start.x(),start.y(),goal.x(),goal.y());
    pq.push({nodes[sid].f, sid});

    // 六种邻居动作（世界系控制，位移与 yaw 解耦）：±x 平移、±y 平移、原地 yaw ±1 bin
    const std::array<int,6> dxs  = {+1, -1,  0,  0, 0, 0};
    const std::array<int,6> dys  = { 0,  0, +1, -1, 0, 0};
    const std::array<int,6> dths = { 0,  0,  0,  0, +1, -1};

    while(!pq.empty()){
      const int id = pq.top().idx; pq.pop();
      Node& cur = nodes[id];
      if (cur.closed) continue;
      cur.closed = true;

      int cx, cy, cth; decode(id, cx, cy, cth);

      // 抵达判定：靠近位置 & 角度接近（bin 环绕距离，35↔0 视为相邻）
      double dx=cx-goal.x(), dy=cy-goal.y();
      const int dth_goal = std::abs(((cth - goal.z() + theta_bins + theta_bins/2)
                                     % theta_bins) - theta_bins/2);
      if (std::sqrt(dx*dx+dy*dy) <= step_pix*1.5 && dth_goal <= 1) {
        return recPath(id);
      }

      // 扩展动作
      for (int k=0;k<6;++k){
        // 平移方向硬约束：dx 必须朝目标 x 方向或为 0（不允许远离目标的 x 位移，
        // 零代价平台上以此保证向目标推进）；dx=0 的平移与旋转不受限
        if (dxs[k] != 0 &&
            (goal.x() > cx) - (goal.x() < cx) != dxs[k]) continue;
        int nth = ( (cth + (int)std::round(dths[k]*yaw_step_bins)) % theta_bins + theta_bins ) % theta_bins;
        // 世界系位移：直接沿 ±x / ±y 平移，与朝向无关
        int nx = cx + static_cast<int>(std::round(dxs[k]*step_pix));
        int ny = cy + static_cast<int>(std::round(dys[k]*step_pix));
        // 运动本身零代价：g 仅由各软惩罚（离障/惯性/姿态）构成
        pushOrRelax(nx,ny,nth,cur.g, id, cx,cy, pq);
      }
    }
    return {}; // 无解
  }
};
