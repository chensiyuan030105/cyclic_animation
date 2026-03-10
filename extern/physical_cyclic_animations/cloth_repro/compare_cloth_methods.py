#!/usr/bin/env python3
"""
Simplified cloth cyclic optimization comparison:
weak baseline vs improved method.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


@dataclass
class ClothConfig:
    nx: int
    ny: int
    spacing: float
    steps: int
    dt: float
    gravity: float
    damping: float
    k_struct: float
    k_shear: float


def make_grid_positions(nx: int, ny: int, spacing: float, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(-(nx - 1) * spacing * 0.5, (nx - 1) * spacing * 0.5, nx, device=device)
    ys = torch.linspace(1.4, 1.4 - (ny - 1) * spacing, ny, device=device)
    xv, yv = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=-1)
    return pos


def idx(x: int, y: int, nx: int) -> int:
    return y * nx + x


def build_edges(cfg: ClothConfig, device: torch.device):
    i_list = []
    j_list = []
    rest = []
    k_list = []

    for y in range(cfg.ny):
        for x in range(cfg.nx):
            a = idx(x, y, cfg.nx)
            if x + 1 < cfg.nx:
                b = idx(x + 1, y, cfg.nx)
                i_list.append(a)
                j_list.append(b)
                rest.append(cfg.spacing)
                k_list.append(cfg.k_struct)
            if y + 1 < cfg.ny:
                b = idx(x, y + 1, cfg.nx)
                i_list.append(a)
                j_list.append(b)
                rest.append(cfg.spacing)
                k_list.append(cfg.k_struct)
            if x + 1 < cfg.nx and y + 1 < cfg.ny:
                b = idx(x + 1, y + 1, cfg.nx)
                i_list.append(a)
                j_list.append(b)
                rest.append(cfg.spacing * np.sqrt(2.0))
                k_list.append(cfg.k_shear)
            if x - 1 >= 0 and y + 1 < cfg.ny:
                b = idx(x - 1, y + 1, cfg.nx)
                i_list.append(a)
                j_list.append(b)
                rest.append(cfg.spacing * np.sqrt(2.0))
                k_list.append(cfg.k_shear)

    i = torch.tensor(i_list, dtype=torch.long, device=device)
    j = torch.tensor(j_list, dtype=torch.long, device=device)
    rest_len = torch.tensor(rest, dtype=torch.float32, device=device)
    k = torch.tensor(k_list, dtype=torch.float32, device=device)
    return i, j, rest_len, k


def load_dataset_sequence(
    dataset_path: str,
    dataset_key: str,
    axis: str,
    start: int,
    length: int,
    downsample: int,
    device: torch.device,
):
    if dataset_path.endswith(".npz"):
        with np.load(dataset_path) as z:
            if dataset_key not in z:
                raise KeyError(f"Key '{dataset_key}' not found in {dataset_path}. Keys: {list(z.keys())}")
            data = z[dataset_key]
    else:
        data = np.load(dataset_path)

    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Expected NxTx3 array, got shape {data.shape}")

    n, t_total, _ = data.shape
    side = int(round(np.sqrt(n)))
    if side * side != n:
        raise ValueError(f"N={n} is not a square number, cannot infer grid topology")
    nx_full = side
    ny_full = side

    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    d0, d1 = axis_map[axis]
    data2 = data[:, :, [d0, d1]].astype(np.float32)

    if start < 0 or start >= t_total - 1:
        raise ValueError(f"Invalid dataset_start={start} for T={t_total}")
    end = t_total if length <= 0 else min(t_total, start + length)
    if end - start < 2:
        raise ValueError(f"Need at least 2 frames, got range [{start}, {end})")
    data2 = data2[:, start:end, :]

    if downsample < 1:
        raise ValueError("dataset_downsample must be >= 1")
    nx = nx_full
    ny = ny_full
    if downsample > 1:
        grid = data2.reshape(ny_full, nx_full, data2.shape[1], 2)
        grid = grid[::downsample, ::downsample, :, :]
        ny, nx = grid.shape[:2]
        data2 = grid.reshape(ny * nx, data2.shape[1], 2)

    frame0 = data2[:, 0, :].reshape(ny, nx, 2)
    dx = float(np.linalg.norm(frame0[:, 1:, :] - frame0[:, :-1, :], axis=2).mean()) if nx > 1 else 1.0
    dy = float(np.linalg.norm(frame0[1:, :, :] - frame0[:-1, :, :], axis=2).mean()) if ny > 1 else 1.0
    spacing = 0.5 * (dx + dy)

    pos0 = torch.tensor(data2[:, 0, :], dtype=torch.float32, device=device)
    target_trace = torch.tensor(np.transpose(data2, (1, 0, 2)), dtype=torch.float32, device=device)

    pin_left_top = idx(0, 0, nx)
    pin_left_bottom = idx(0, ny - 1, nx)
    info = {
        "dataset_path": dataset_path,
        "dataset_key": dataset_key if dataset_path.endswith(".npz") else "",
        "axis": axis,
        "start_frame": start,
        "end_frame_exclusive": start + target_trace.shape[0],
        "n_points_full": int(n),
        "n_points_used": int(nx * ny),
        "nx_full": int(nx_full),
        "ny_full": int(ny_full),
        "nx_used": int(nx),
        "ny_used": int(ny),
        "downsample": int(downsample),
        "estimated_spacing": float(spacing),
        "pinned_indices": [int(pin_left_top), int(pin_left_bottom)],
    }
    return pos0, target_trace, nx, ny, spacing, (pin_left_top, pin_left_bottom), info


def rollout(
    pos0: torch.Tensor,
    vel0: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    rest_len: torch.Tensor,
    k_edge: torch.Tensor,
    pinned_mask: torch.Tensor,
    center_idx: int,
    cfg: ClothConfig,
):
    pos = pos0
    vel = vel0
    center_trace = [pos[center_idx]]
    pos_trace = [pos]

    for _ in range(cfg.steps):
        force = torch.zeros_like(pos)
        force[:, 1] -= cfg.gravity

        diff = pos[j] - pos[i]
        dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)
        dir_vec = diff / dist[:, None]
        f_spring = k_edge[:, None] * (dist - rest_len)[:, None] * dir_vec

        force = force.index_add(0, i, f_spring)
        force = force.index_add(0, j, -f_spring)

        force = force - cfg.damping * vel

        vel = vel + cfg.dt * force
        pos = pos + cfg.dt * vel

        pos = torch.where(pinned_mask[:, None], pos0, pos)
        vel = torch.where(pinned_mask[:, None], torch.zeros_like(vel), vel)

        center_trace.append(pos[center_idx])
        pos_trace.append(pos)

    return pos, vel, torch.stack(center_trace, dim=0), torch.stack(pos_trace, dim=0)


def save_history(path: str, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["iter", "loss", "closure_pos", "closure_vel", "trajectory_mse"]
        )
        writer.writeheader()
        writer.writerows(rows)


def draw_mesh(ax, pos_np: np.ndarray, i_np: np.ndarray, j_np: np.ndarray, color: str, alpha: float, lw: float):
    for a, b in zip(i_np, j_np):
        xa, ya = pos_np[a]
        xb, yb = pos_np[b]
        ax.plot([xa, xb], [ya, yb], color=color, alpha=alpha, linewidth=lw)


def _compute_bounds(traces: Iterable[np.ndarray], pad_ratio: float = 0.05):
    pts = np.concatenate([t.reshape(-1, 2) for t in traces], axis=0)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)
    xmin -= dx * pad_ratio
    xmax += dx * pad_ratio
    ymin -= dy * pad_ratio
    ymax += dy * pad_ratio
    return float(xmin), float(xmax), float(ymin), float(ymax)


def _world_to_pixel(pos: np.ndarray, bounds, w: int, h: int, margin: int):
    xmin, xmax, ymin, ymax = bounds
    sx = (w - 2 * margin) / max(1e-6, (xmax - xmin))
    sy = (h - 2 * margin) / max(1e-6, (ymax - ymin))
    x = margin + (pos[:, 0] - xmin) * sx
    y = h - margin - (pos[:, 1] - ymin) * sy
    return np.stack([x, y], axis=-1)


def _draw_mesh_image(
    pos: np.ndarray,
    i_np: np.ndarray,
    j_np: np.ndarray,
    bounds,
    size=(640, 480),
    margin: int = 24,
    line_color=(30, 120, 220),
    bg=(255, 255, 255),
    title: str = "",
):
    w, h = size
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    pxy = _world_to_pixel(pos, bounds, w, h, margin)
    for a, b in zip(i_np, j_np):
        xa, ya = pxy[a]
        xb, yb = pxy[b]
        draw.line([(float(xa), float(ya)), (float(xb), float(yb))], fill=line_color, width=1)
    if title:
        draw.text((8, 8), title, fill=(0, 0, 0))
    return img


def save_sequence_gif(
    out_path: str,
    trace: np.ndarray,
    i_np: np.ndarray,
    j_np: np.ndarray,
    bounds,
    fps: int,
    title_prefix: str,
):
    duration = int(round(1000.0 / max(1, fps)))
    frames = []
    for t in range(trace.shape[0]):
        title = f"{title_prefix}  frame {t}"
        frames.append(_draw_mesh_image(trace[t], i_np, j_np, bounds=bounds, title=title))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )


def save_triplet_sequence_gif(
    out_path: str,
    weak_trace: np.ndarray,
    improved_trace: np.ndarray,
    i_np: np.ndarray,
    j_np: np.ndarray,
    bounds,
    fps: int,
    target_trace: np.ndarray | None = None,
):
    duration = int(round(1000.0 / max(1, fps)))
    frames = []
    n = weak_trace.shape[0]
    for t in range(n):
        if target_trace is not None:
            left = _draw_mesh_image(
                target_trace[t], i_np, j_np, bounds=bounds, size=(480, 420), title=f"target  frame {t}"
            )
        else:
            left = Image.new("RGB", (480, 420), (245, 245, 245))
            d = ImageDraw.Draw(left)
            d.text((12, 12), f"no target trace  frame {t}", fill=(0, 0, 0))
        mid = _draw_mesh_image(weak_trace[t], i_np, j_np, bounds=bounds, size=(480, 420), title=f"weak  frame {t}")
        right = _draw_mesh_image(
            improved_trace[t], i_np, j_np, bounds=bounds, size=(480, 420), title=f"improved  frame {t}"
        )
        canvas = Image.new("RGB", (left.width + mid.width + right.width + 16, 420), (255, 255, 255))
        canvas.paste(left, (0, 0))
        canvas.paste(mid, (left.width + 8, 0))
        canvas.paste(right, (left.width + mid.width + 16, 0))
        frames.append(canvas)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )


def run_method(
    method: str,
    cfg: ClothConfig,
    pos0: torch.Tensor,
    vel_init: torch.Tensor,
    pinned_mask: torch.Tensor,
    free_mask: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    rest_len: torch.Tensor,
    k_edge: torch.Tensor,
    center_idx: int,
    iters: int,
    lr: float,
    target_trace: torch.Tensor | None = None,
    match_weight: float = 0.0,
):
    vel_param = torch.nn.Parameter(vel_init.clone())
    if method == "weak":
        optimizer = torch.optim.SGD([vel_param], lr=lr)
    else:
        optimizer = torch.optim.Adam([vel_param], lr=lr)

    history = []

    free_count = free_mask.sum()
    free_mask_f = free_mask.float()

    for it in range(1, iters + 1):
        optimizer.zero_grad()

        if method == "weak":
            v0 = vel_param
        else:
            mean_v = (vel_param * free_mask_f[:, None]).sum(dim=0) / free_count
            v0 = vel_param - mean_v[None, :] * free_mask_f[:, None]

        pos_t, vel_t, center_trace, pos_trace = rollout(
            pos0=pos0,
            vel0=v0,
            i=i,
            j=j,
            rest_len=rest_len,
            k_edge=k_edge,
            pinned_mask=pinned_mask,
            center_idx=center_idx,
            cfg=cfg,
        )

        closure_pos = ((pos_t[free_mask] - pos0[free_mask]) ** 2).mean()
        closure_vel = ((vel_t[free_mask] - v0[free_mask]) ** 2).mean()
        trajectory_mse = torch.tensor(0.0, dtype=pos0.dtype, device=pos0.device)
        if target_trace is not None:
            trajectory_mse = ((pos_trace[:, free_mask] - target_trace[:, free_mask]) ** 2).mean()

        if method == "weak":
            loss = closure_pos + match_weight * trajectory_mse
        else:
            vel_reg = (v0[free_mask] ** 2).mean()
            loss = closure_pos + 0.5 * closure_vel + 1e-3 * vel_reg + match_weight * trajectory_mse

        loss.backward()
        optimizer.step()

        history.append(
            {
                "iter": it,
                "loss": float(loss.item()),
                "closure_pos": float(closure_pos.item()),
                "closure_vel": float(closure_vel.item()),
                "trajectory_mse": float(trajectory_mse.item()),
            }
        )

    with torch.no_grad():
        if method == "weak":
            v0 = vel_param
        else:
            mean_v = (vel_param * free_mask_f[:, None]).sum(dim=0) / free_count
            v0 = vel_param - mean_v[None, :] * free_mask_f[:, None]
        pos_t, vel_t, center_trace, pos_trace = rollout(
            pos0=pos0,
            vel0=v0,
            i=i,
            j=j,
            rest_len=rest_len,
            k_edge=k_edge,
            pinned_mask=pinned_mask,
            center_idx=center_idx,
            cfg=cfg,
        )

        metrics = {
            "closure_pos_rmse": float(torch.sqrt(((pos_t[free_mask] - pos0[free_mask]) ** 2).mean()).item()),
            "closure_vel_rmse": float(torch.sqrt(((vel_t[free_mask] - v0[free_mask]) ** 2).mean()).item()),
            "center_loop_closure": float(torch.norm(center_trace[-1] - center_trace[0]).item()),
        }
        if target_trace is not None:
            metrics["trajectory_rmse"] = float(
                torch.sqrt(((pos_trace[:, free_mask] - target_trace[:, free_mask]) ** 2).mean()).item()
            )

    return {
        "history": history,
        "pos_final": pos_t.detach().cpu().numpy(),
        "center_trace": center_trace.detach().cpu().numpy(),
        "pos_trace": pos_trace.detach().cpu().numpy(),
        "metrics": metrics,
    }


def make_panel(loss_png: str, state_png: str, out_png: str):
    left = Image.open(loss_png).convert("RGB")
    right = Image.open(state_png).convert("RGB")
    h = max(left.height, right.height)
    left = ImageOps.pad(left, (left.width, h), color="white")
    right = ImageOps.pad(right, (right.width, h), color="white")
    pad = 30
    canvas = Image.new("RGB", (left.width + right.width + 3 * pad, h + 2 * pad), "white")
    canvas.paste(left, (pad, pad))
    canvas.paste(right, (left.width + 2 * pad, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), "A. Loss Comparison", fill="black")
    draw.text((left.width + 2 * pad, 8), "B. Cloth State/Trajectory Comparison", fill="black")
    canvas.save(out_png)


def parse_args():
    p = argparse.ArgumentParser(description="Compare weak vs improved cloth cyclic optimization methods.")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--nx", type=int, default=6)
    p.add_argument("--ny", type=int, default=6)
    p.add_argument("--spacing", type=float, default=0.18)
    p.add_argument("--gravity", type=float, default=0.0)
    p.add_argument("--damping", type=float, default=0.08)
    p.add_argument("--k_struct", type=float, default=180.0)
    p.add_argument("--k_shear", type=float, default=100.0)
    p.add_argument("--noise_scale", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--lr_weak", type=float, default=0.03)
    p.add_argument("--lr_improved", type=float, default=0.05)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out_dir", type=str, default="outputs/compare_default")
    p.add_argument("--dataset", type=str, default="", help="Path to NxTx3 .npy/.npz cloth trajectory")
    p.add_argument("--dataset_key", type=str, default="position_nxt3", help="Key used when --dataset is npz")
    p.add_argument("--dataset_axis", choices=["xy", "xz", "yz"], default="xz")
    p.add_argument("--dataset_start", type=int, default=0)
    p.add_argument(
        "--dataset_len",
        type=int,
        default=0,
        help="Frames to load from dataset (0 means use all available from start)",
    )
    p.add_argument("--dataset_downsample", type=int, default=1, help="Grid downsample stride for dataset mode")
    p.add_argument("--match_weight", type=float, default=0.2, help="Weight for trajectory matching loss")
    p.add_argument("--save_sequence", type=int, default=1, help="Whether to export sequence GIFs")
    p.add_argument("--sequence_fps", type=int, default=20)
    p.add_argument("--sequence_stride", type=int, default=1, help="Frame stride used for sequence export")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    dataset_info = None
    target_trace = None
    pin_pair = None

    if args.dataset:
        pos0, target_trace, nx, ny, spacing, pin_pair, dataset_info = load_dataset_sequence(
            dataset_path=args.dataset,
            dataset_key=args.dataset_key,
            axis=args.dataset_axis,
            start=args.dataset_start,
            length=args.dataset_len,
            downsample=args.dataset_downsample,
            device=device,
        )
        steps = min(args.steps, target_trace.shape[0] - 1)
        target_trace = target_trace[: steps + 1]
        cfg = ClothConfig(
            nx=nx,
            ny=ny,
            spacing=spacing,
            steps=steps,
            dt=args.dt,
            gravity=0.0,
            damping=args.damping,
            k_struct=args.k_struct,
            k_shear=args.k_shear,
        )
        vel_base = (target_trace[1] - target_trace[0]) / cfg.dt
        vel_init = vel_base + args.noise_scale * torch.randn_like(vel_base)
    else:
        cfg = ClothConfig(
            nx=args.nx,
            ny=args.ny,
            spacing=args.spacing,
            steps=args.steps,
            dt=args.dt,
            gravity=args.gravity,
            damping=args.damping,
            k_struct=args.k_struct,
            k_shear=args.k_shear,
        )
        pos0 = make_grid_positions(cfg.nx, cfg.ny, cfg.spacing, device)
        vel_base = torch.zeros_like(pos0)
        vel_init = vel_base + args.noise_scale * torch.randn_like(vel_base)

    pinned_mask = torch.zeros(pos0.shape[0], dtype=torch.bool, device=device)
    if pin_pair is None:
        pinned_mask[idx(0, 0, cfg.nx)] = True
        pinned_mask[idx(cfg.nx - 1, 0, cfg.nx)] = True
    else:
        pinned_mask[pin_pair[0]] = True
        pinned_mask[pin_pair[1]] = True
    free_mask = ~pinned_mask

    i, j, rest_len, k_edge = build_edges(cfg, device)
    center_idx = idx(cfg.nx // 2, cfg.ny // 2, cfg.nx)

    weak = run_method(
        method="weak",
        cfg=cfg,
        pos0=pos0,
        vel_init=vel_init,
        pinned_mask=pinned_mask,
        free_mask=free_mask,
        i=i,
        j=j,
        rest_len=rest_len,
        k_edge=k_edge,
        center_idx=center_idx,
        iters=args.iters,
        lr=args.lr_weak,
        target_trace=target_trace,
        match_weight=args.match_weight if target_trace is not None else 0.0,
    )
    improved = run_method(
        method="improved",
        cfg=cfg,
        pos0=pos0,
        vel_init=vel_init,
        pinned_mask=pinned_mask,
        free_mask=free_mask,
        i=i,
        j=j,
        rest_len=rest_len,
        k_edge=k_edge,
        center_idx=center_idx,
        iters=args.iters,
        lr=args.lr_improved,
        target_trace=target_trace,
        match_weight=args.match_weight if target_trace is not None else 0.0,
    )

    save_history(os.path.join(args.out_dir, "weak_history.csv"), weak["history"])
    save_history(os.path.join(args.out_dir, "improved_history.csv"), improved["history"])

    summary = {
        "config": cfg.__dict__,
        "weak": weak["metrics"],
        "improved": improved["metrics"],
        "dataset": dataset_info,
    }
    summary_path = os.path.join(args.out_dir, "summary_metrics.json")

    plt.figure(figsize=(7.5, 4.8))
    xw = [d["iter"] for d in weak["history"]]
    yw = [d["closure_pos"] for d in weak["history"]]
    xi = [d["iter"] for d in improved["history"]]
    yi = [d["closure_pos"] for d in improved["history"]]
    plt.semilogy(xw, yw, label="Weak baseline (SGD, closure only)", linewidth=2.0)
    plt.semilogy(xi, yi, label="Improved (projected + regularized)", linewidth=2.2)
    plt.xlabel("Optimization Iteration")
    plt.ylabel("Position Closure MSE (log)")
    plt.title("Cloth Cyclic Optimization: Convergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_png = os.path.join(args.out_dir, "figure_loss_compare.png")
    plt.savefig(loss_png, dpi=180)
    plt.close()

    i_np = i.detach().cpu().numpy()
    j_np = j.detach().cpu().numpy()
    pos0_np = pos0.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, res, title, col in [
        (axes[0], weak, "Weak baseline", "#d62728"),
        (axes[1], improved, "Improved method", "#2ca02c"),
    ]:
        draw_mesh(ax, pos0_np, i_np, j_np, color="#808080", alpha=0.35, lw=0.8)
        draw_mesh(ax, res["pos_final"], i_np, j_np, color=col, alpha=0.95, lw=1.2)
        tr = res["center_trace"]
        ax.plot(tr[:, 0], tr[:, 1], color="#1f77b4", linewidth=1.8, linestyle="--", label="Center trajectory")
        ax.scatter(tr[0, 0], tr[0, 1], color="#1f77b4", s=28, marker="o")
        ax.scatter(tr[-1, 0], tr[-1, 1], color="#1f77b4", s=28, marker="x")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower left")

    fig.suptitle("Cloth: Initial mesh (gray) vs final mesh (colored)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    state_png = os.path.join(args.out_dir, "figure_state_compare.png")
    fig.savefig(state_png, dpi=180)
    plt.close(fig)

    panel_png = os.path.join(args.out_dir, "figure_paper_style_panel.png")
    make_panel(loss_png, state_png, panel_png)

    sequence_outputs = []
    if args.save_sequence:
        seq_stride = max(1, int(args.sequence_stride))
        weak_trace = weak["pos_trace"][::seq_stride]
        improved_trace = improved["pos_trace"][::seq_stride]
        target_np = target_trace.detach().cpu().numpy()[::seq_stride] if target_trace is not None else None

        traces_for_bounds = [weak_trace, improved_trace]
        if target_np is not None:
            traces_for_bounds.append(target_np)
        bounds = _compute_bounds(traces_for_bounds)

        weak_gif = os.path.join(args.out_dir, "sequence_weak.gif")
        save_sequence_gif(
            out_path=weak_gif,
            trace=weak_trace,
            i_np=i_np,
            j_np=j_np,
            bounds=bounds,
            fps=args.sequence_fps,
            title_prefix="weak",
        )
        sequence_outputs.append(weak_gif)

        improved_gif = os.path.join(args.out_dir, "sequence_improved.gif")
        save_sequence_gif(
            out_path=improved_gif,
            trace=improved_trace,
            i_np=i_np,
            j_np=j_np,
            bounds=bounds,
            fps=args.sequence_fps,
            title_prefix="improved",
        )
        sequence_outputs.append(improved_gif)

        if target_np is not None:
            target_gif = os.path.join(args.out_dir, "sequence_target.gif")
            save_sequence_gif(
                out_path=target_gif,
                trace=target_np,
                i_np=i_np,
                j_np=j_np,
                bounds=bounds,
                fps=args.sequence_fps,
                title_prefix="target",
            )
            sequence_outputs.append(target_gif)

        triplet_gif = os.path.join(args.out_dir, "sequence_triplet.gif")
        save_triplet_sequence_gif(
            out_path=triplet_gif,
            weak_trace=weak_trace,
            improved_trace=improved_trace,
            i_np=i_np,
            j_np=j_np,
            bounds=bounds,
            fps=args.sequence_fps,
            target_trace=target_np,
        )
        sequence_outputs.append(triplet_gif)

        summary["sequence"] = {
            "enabled": True,
            "fps": int(args.sequence_fps),
            "stride": int(seq_stride),
            "n_frames": int(weak_trace.shape[0]),
            "files": sequence_outputs,
        }
    else:
        summary["sequence"] = {"enabled": False}

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done. Outputs:")
    print(f"- {loss_png}")
    print(f"- {state_png}")
    print(f"- {panel_png}")
    for p in sequence_outputs:
        print(f"- {p}")
    print(f"- {summary_path}")
    print("\nSummary metrics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
