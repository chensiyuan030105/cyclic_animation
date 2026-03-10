# Cloth Dataset 接入方案与执行记录（2026-03-10）

## 背景
- 输入数据：`/workspace/cyclic_animation/dataset/cloth_flag_flutter_square5K_T200_NxTx3.npy`
- 数据形状：`(N,T,3)=(5041,200,3)`，即 `71x71` 网格、200 帧。
- 目标：用该数据驱动 `physical_cyclic_animations/cloth_repro` 跑 weak vs improved 对比。

## 接入思路
1. 在 `compare_cloth_methods.py` 增加 dataset 模式（不破坏原 toy 模式）。
2. 支持读取 `.npy/.npz`，npz 默认 key `position_nxt3`。
3. 支持 3D->2D 轴投影（`xy/xz/yz`），此任务优先用 `xz`。
4. 自动从 `N` 推断方格拓扑（`sqrt(N)`），并估计网格 spacing。
5. 支持空间降采样（`dataset_downsample`）以控制计算量。
6. 仍沿用 mass-spring rollout；在损失中加入 trajectory matching 项：
   - `loss = closure + ... + match_weight * trajectory_mse`
7. 夹持点策略（dataset 模式）：固定左边界两个角点（left-top / left-bottom）。

## 参数策略（本次 smoke）
- `dataset_axis = xz`
- `dataset_start = 75`
- `dataset_len = 60`（可用 60 帧窗口）
- `steps = 50`（使用窗口前 51 帧）
- `dataset_downsample = 3`（71x71 -> 24x24）
- `iters = 60`
- `match_weight = 0.2`

以上配置优先保证先跑通，并在分钟级拿到可比指标。

## 运行命令
```bash
cd /workspace/cyclic_animation/extern/physical_cyclic_animations/cloth_repro
python3 compare_cloth_methods.py \
  --dataset /workspace/cyclic_animation/dataset/cloth_flag_flutter_square5K_T200_NxTx3.npy \
  --dataset_axis xz \
  --dataset_start 75 \
  --dataset_len 60 \
  --dataset_downsample 3 \
  --steps 50 \
  --iters 60 \
  --dt 0.02 \
  --match_weight 0.2 \
  --out_dir outputs/dataset_smoke_20260310
```

## 运行结果
- 运行成功，输出目录：
  - `outputs/dataset_smoke_20260310`
- 主要产物：
  - `figure_loss_compare.png`
  - `figure_state_compare.png`
  - `figure_paper_style_panel.png`
  - `weak_history.csv`
  - `improved_history.csv`
  - `summary_metrics.json`

### 指标摘录（summary_metrics.json）
- 配置（自动推断后）：
  - `nx=24, ny=24`（由 `71x71` 以 stride=3 降采样）
  - `steps=50, dt=0.02`
  - `estimated_spacing=0.030716`
- Weak：
  - `closure_pos_rmse=3.4041`
  - `closure_vel_rmse=6.2786`
  - `center_loop_closure=5.0398`
  - `trajectory_rmse=1.6126`
- Improved：
  - `closure_pos_rmse=3.2781`
  - `closure_vel_rmse=6.2073`
  - `center_loop_closure=4.8398`
  - `trajectory_rmse=1.5028`

### 初步结论
- 在当前 smoke 配置下，`improved` 相对 `weak` 在四项指标上均有小幅提升。
- 但整体误差仍偏大，说明“数据窗口 + 动力学参数 + 损失权重”还需继续调参。

### 结果一句话总结
- 这轮已经“跑通并优于 baseline”：`improved` 相比 `weak` 的提升约为：
  - `trajectory_rmse`: `1.6126 -> 1.5028`（约 **6.8%** 改善）
  - `center_loop_closure`: `5.0398 -> 4.8398`（约 **4.0%** 改善）
  - `closure_pos_rmse`: `3.4041 -> 3.2781`（约 **3.7%** 改善）
  - `closure_vel_rmse`: `6.2786 -> 6.2073`（约 **1.1%** 改善）

### 下一步收敛方向
1. 调整窗口：尝试 `dataset_start=100~140`，优先选更平稳段。
2. 提高轨迹约束：增大 `match_weight`（如 `0.5~1.0`）并观察 `trajectory_rmse`。
3. 先用 `dataset_downsample=4/5` 做粗调，再回到 `3`。
4. 在 `xz` 固定后，对 `k_struct/k_shear/damping` 做小网格搜索。
