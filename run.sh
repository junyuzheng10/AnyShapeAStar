#!/usr/bin/env bash
# AnyShapeAStar 一键启动脚本：构建 -> 运行 -> 合成 GIF 动画
set -e
cd "$(dirname "$0")"

echo "==> [1/3] 构建项目（增量编译）"
mkdir -p build
cd build
cmake .. > /dev/null
make -j2

echo "==> [2/3] 运行 demo（关闭窗口或按任意键继续）"
./anyshape_astar_demo

echo "==> [3/3] 合成 GIF 动画（PIL，替代裁剪版 ffmpeg）"
python3 - <<'EOF'
from PIL import Image
import glob

# 描边版 + 点云版 两套帧各自合成
for frames_dir, out_gif in [('anim_frames', 'path_animation.gif'),
                            ('anim_frames_cloud', 'path_animation_cloud.gif')]:
    files = sorted(glob.glob(f'{frames_dir}/frame_*.png'))
    if not files:
        print(f'[gif] 未找到 {frames_dir} 帧，跳过 GIF 合成')
        continue
    imgs = [Image.open(f).convert('P', palette=Image.ADAPTIVE) for f in files]
    imgs[0].save(out_gif, save_all=True,
                 append_images=imgs[1:], duration=33, loop=0)
    print(f'[gif] 已生成 {len(imgs)} 帧 -> build/{out_gif}')
EOF

echo "==> 完成！动画位置：$(pwd)/path_animation.gif 与 $(pwd)/path_animation_cloud.gif"
