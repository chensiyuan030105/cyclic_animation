#!/usr/bin/env python3
"""
DMD-ROM cloth baseline on full-resolution NxTx3 trajectories.

Key points:
- No mesh downsampling (keeps all N points).
- ROM reduction happens in state space via PCA.
- Delay-DMD in latent space with periodic spectrum projection.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class DMDConfig:
    dt: float
    pca_rank: int
    pca_energy: float
    delay: int
    dmd_rank: int
    period: int
    radius_eps: float
    rollout_frames: int
    axis: str
    gif_fps: int
    gif_stride: int
    project_periodic: bool


def load_positions(dataset_path: str, dataset_key: str) -> np.ndarray:
    if dataset_path.endswith(".npz"):
        with np.load(dataset_path) as z:
            if dataset_key not in z:
                raise KeyError(f"Key '{dataset_key}' not found in {dataset_path}. Keys: {list(z.keys())}")
            arr = z[dataset_key]
    else:
        arr = np.load(dataset_path)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected array with shape NxTx3 or TxNx3, got {arr.shape}")

    # Auto layout: this dataset is typically NxTx3 with N >> T.
    if arr.shape[0] > arr.shape[1]:
        arr = np.transpose(arr, (1, 0, 2))
    # now TxNx3
    return arr.astype(np.float64)


def compute_velocity(positions_tnx3: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(positions_tnx3)
    vel[0] = (positions_tnx3[1] - positions_tnx3[0]) / dt
    vel[-1] = (positions_tnx3[-1] - positions_tnx3[-2]) / dt
    vel[1:-1] = (positions_tnx3[2:] - positions_tnx3[:-2]) / (2.0 * dt)
    return vel


def build_point_features(n_points: int) -> np.ndarray:
    side = int(round(np.sqrt(n_points)))
    if side * side != n_points:
        # Fallback for non-square topology: mass + index encoding.
        idx = np.arange(n_points, dtype=np.float64)
        mass = np.ones_like(idx)
        feats = np.stack([mass, idx / max(1, n_points - 1)], axis=1)
        return feats.astype(np.float32)

    row = np.repeat(np.arange(side), side)
    col = np.tile(np.arange(side), side)
    mass = np.ones(n_points, dtype=np.float64)
    boundary = ((row == 0) | (row == side - 1) | (col == 0) | (col == side - 1)).astype(np.float64)
    top_edge = (row == 0).astype(np.float64)
    u = col / max(1, side - 1)
    v = row / max(1, side - 1)
    feats = np.stack([mass, boundary, top_edge, u, v], axis=1)
    return feats.astype(np.float32)


def estimate_period(series: np.ndarray) -> int:
    x = series - series.mean()
    fft = np.fft.rfft(x)
    amp = np.abs(fft)
    if amp.shape[0] <= 1:
        return 2
    amp[0] = 0.0
    idx = int(np.argmax(amp))
    if idx <= 0:
        return max(2, len(series) // 2)
    p = int(round(len(series) / idx))
    return int(np.clip(p, 2, max(2, len(series) - 1)))


def fit_pca(states_txd: np.ndarray, rank: int, energy: float):
    mean = states_txd.mean(axis=0, keepdims=True)
    x = states_txd - mean
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    if rank > 0:
        r = min(rank, s.shape[0])
    else:
        e = np.cumsum(s * s) / np.maximum(1e-12, np.sum(s * s))
        r = int(np.searchsorted(e, energy) + 1)
        r = min(r, s.shape[0])
    z = u[:, :r] * s[:r]
    return mean, vt[:r], z, s


def make_delay_embedding(z_txr: np.ndarray, delay: int) -> np.ndarray:
    t, r = z_txr.shape
    if delay < 1 or delay >= t:
        raise ValueError(f"delay must satisfy 1 <= delay < T, got delay={delay}, T={t}")
    n = t - delay + 1
    y = np.zeros((n, delay * r), dtype=np.float64)
    for i in range(n):
        y[i] = z_txr[i : i + delay].reshape(-1)
    return y


def fit_delay_dmd(y_nxd: np.ndarray, rank: int):
    # y[k+1] = A y[k]
    x = y_nxd[:-1].T
    xp = y_nxd[1:].T
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    if rank > 0:
        r = min(rank, s.shape[0])
    else:
        r = s.shape[0]
    ur = u[:, :r]
    sr = s[:r]
    vr = vh[:r, :].T

    atilde = ur.T @ xp @ vr
    atilde = atilde / sr[None, :]

    eigvals, w = np.linalg.eig(atilde)
    phi = xp @ vr
    phi = phi / sr[None, :]
    phi = phi @ w
    return eigvals, phi


def project_eigs_to_period(eigvals: np.ndarray, period: int, radius_eps: float):
    theta = np.angle(eigvals)
    k = np.round(theta * period / (2.0 * np.pi))
    theta_proj = 2.0 * np.pi * k / period
    if radius_eps <= 0.0:
        rho_proj = np.ones_like(theta_proj)
    else:
        rho = np.abs(eigvals)
        rho_proj = np.clip(rho, 1.0 - radius_eps, 1.0 + radius_eps)
    return rho_proj * np.exp(1j * theta_proj)


def rollout_delay_dmd(
    phi: np.ndarray,
    eigvals_proj: np.ndarray,
    y0: np.ndarray,
    z_seed_txr: np.ndarray,
    delay: int,
    rollout_frames: int,
) -> np.ndarray:
    r = z_seed_txr.shape[1]
    if rollout_frames < delay:
        raise ValueError(f"rollout_frames must be >= delay, got {rollout_frames} < {delay}")

    z_out = np.zeros((rollout_frames, r), dtype=np.float64)
    z_out[: delay - 1] = z_seed_txr[: delay - 1]

    b = np.linalg.pinv(phi) @ y0
    n_states = rollout_frames - delay + 1
    for n in range(n_states):
        y_n = phi @ ((eigvals_proj**n) * b)
        y_n = np.real(y_n)
        frame_idx = n + delay - 1
        z_out[frame_idx] = y_n[-r:]
    return z_out


def _compute_bounds(traces: list[np.ndarray], axis_idx: tuple[int, int], pad_ratio: float = 0.06):
    pts = np.concatenate([t[:, :, [axis_idx[0], axis_idx[1]]].reshape(-1, 2) for t in traces], axis=0)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)
    xmin -= dx * pad_ratio
    xmax += dx * pad_ratio
    ymin -= dy * pad_ratio
    ymax += dy * pad_ratio
    return float(xmin), float(xmax), float(ymin), float(ymax)


def _world_to_pixel(pos_nx2: np.ndarray, bounds, w: int, h: int, margin: int):
    xmin, xmax, ymin, ymax = bounds
    sx = (w - 2 * margin) / max(1e-6, xmax - xmin)
    sy = (h - 2 * margin) / max(1e-6, ymax - ymin)
    x = margin + (pos_nx2[:, 0] - xmin) * sx
    y = h - margin - (pos_nx2[:, 1] - ymin) * sy
    return np.stack([x, y], axis=-1)


def _render_points(pos_nx3: np.ndarray, axis_idx: tuple[int, int], bounds, color, title: str):
    w, h = 480, 420
    margin = 20
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    pos2 = pos_nx3[:, [axis_idx[0], axis_idx[1]]]
    pxy = _world_to_pixel(pos2, bounds, w, h, margin)
    xi = np.clip(np.round(pxy[:, 0]).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(pxy[:, 1]).astype(np.int32), 0, h - 1)
    img[yi, xi, :] = np.array(color, dtype=np.uint8)[None, :]
    pil = Image.fromarray(img, mode="RGB")
    draw = ImageDraw.Draw(pil)
    draw.text((8, 8), title, fill=(0, 0, 0))
    return pil


def _render_overlay(target_nx3: np.ndarray, pred_nx3: np.ndarray, axis_idx: tuple[int, int], bounds, title: str):
    w, h = 480, 420
    margin = 20
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    t2 = _world_to_pixel(target_nx3[:, [axis_idx[0], axis_idx[1]]], bounds, w, h, margin)
    p2 = _world_to_pixel(pred_nx3[:, [axis_idx[0], axis_idx[1]]], bounds, w, h, margin)
    tx = np.clip(np.round(t2[:, 0]).astype(np.int32), 0, w - 1)
    ty = np.clip(np.round(t2[:, 1]).astype(np.int32), 0, h - 1)
    px = np.clip(np.round(p2[:, 0]).astype(np.int32), 0, w - 1)
    py = np.clip(np.round(p2[:, 1]).astype(np.int32), 0, h - 1)
    img[ty, tx, :] = np.array([40, 120, 220], dtype=np.uint8)[None, :]
    img[py, px, :] = np.array([220, 70, 70], dtype=np.uint8)[None, :]
    pil = Image.fromarray(img, mode="RGB")
    draw = ImageDraw.Draw(pil)
    draw.text((8, 8), title, fill=(0, 0, 0))
    return pil


def save_triplet_gif(
    out_path: str,
    target_tnx3: np.ndarray,
    pred_tnx3: np.ndarray,
    axis: str,
    fps: int,
    stride: int,
):
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    axis_idx = axis_map[axis]
    trace_t = target_tnx3[::stride]
    trace_p = pred_tnx3[::stride]
    bounds = _compute_bounds([trace_t, trace_p], axis_idx=axis_idx)
    duration = int(round(1000.0 / max(1, fps)))

    frames = []
    n = min(trace_t.shape[0], trace_p.shape[0])
    for t in range(n):
        left = _render_points(trace_t[t], axis_idx, bounds, color=(40, 120, 220), title=f"target  frame {t}")
        mid = _render_points(trace_p[t], axis_idx, bounds, color=(220, 70, 70), title=f"dmd-rom  frame {t}")
        right = _render_overlay(trace_t[t], trace_p[t], axis_idx, bounds, title=f"overlay  frame {t}")
        canvas = Image.new("RGB", (1456, 420), (255, 255, 255))
        canvas.paste(left, (0, 0))
        canvas.paste(mid, (488, 0))
        canvas.paste(right, (976, 0))
        frames.append(canvas)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )


def save_single_gif(out_path: str, trace_tnx3: np.ndarray, axis: str, fps: int, stride: int, title: str, color):
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    axis_idx = axis_map[axis]
    trace = trace_tnx3[::stride]
    bounds = _compute_bounds([trace], axis_idx=axis_idx)
    duration = int(round(1000.0 / max(1, fps)))

    frames = []
    for t in range(trace.shape[0]):
        frames.append(_render_points(trace[t], axis_idx, bounds, color=color, title=f"{title}  frame {t}"))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Full-resolution cloth DMD-ROM baseline.")
    p.add_argument(
        "--dataset",
        type=str,
        default="/workspace/cyclic_animation/dataset/cloth_flag_flutter_square5K_T200_NxTx3.npy",
    )
    p.add_argument("--dataset_key", type=str, default="position_nxt3")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--train_frames", type=int, default=0, help="0 means use all frames from start")
    p.add_argument("--rollout_frames", type=int, default=100)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--pca_rank", type=int, default=64, help="0 means rank by --pca_energy")
    p.add_argument("--pca_energy", type=float, default=0.999)
    p.add_argument("--delay", type=int, default=20)
    p.add_argument("--dmd_rank", type=int, default=120, help="0 means full rank in delay subspace")
    p.add_argument("--period", type=int, default=0, help="0 means auto-estimate from latent signal")
    p.add_argument("--radius_eps", type=float, default=0.0, help="0 means project all |lambda| to 1")
    p.add_argument("--project_periodic", type=int, default=1, help="1: periodic spectrum projection, 0: raw DMD")
    p.add_argument("--reconstruct_train", type=int, default=0, help="1: rollout length equals train_frames")
    p.add_argument("--axis", choices=["xy", "xz", "yz"], default="xz")
    p.add_argument("--save_sequence", type=int, default=1)
    p.add_argument("--gif_fps", type=int, default=20)
    p.add_argument("--gif_stride", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="outputs/dmd_rom_v1")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = DMDConfig(
        dt=args.dt,
        pca_rank=args.pca_rank,
        pca_energy=args.pca_energy,
        delay=args.delay,
        dmd_rank=args.dmd_rank,
        period=args.period,
        radius_eps=args.radius_eps,
        rollout_frames=args.rollout_frames,
        axis=args.axis,
        gif_fps=args.gif_fps,
        gif_stride=max(1, args.gif_stride),
        project_periodic=bool(args.project_periodic),
    )

    pos_all = load_positions(args.dataset, args.dataset_key)
    t_total, n_points, _ = pos_all.shape

    start = int(np.clip(args.start, 0, max(0, t_total - 2)))
    end = t_total if args.train_frames <= 0 else min(t_total, start + args.train_frames)
    pos = pos_all[start:end]
    if pos.shape[0] < 4:
        raise ValueError(f"Need at least 4 frames in train slice, got {pos.shape[0]}")

    vel = compute_velocity(pos, cfg.dt)
    states = np.concatenate([pos.reshape(pos.shape[0], -1), vel.reshape(vel.shape[0], -1)], axis=1)

    mean, pca_basis, z, svals = fit_pca(states, rank=cfg.pca_rank, energy=cfg.pca_energy)
    y = make_delay_embedding(z, delay=cfg.delay)
    eigvals, phi = fit_delay_dmd(y, rank=cfg.dmd_rank)

    if cfg.period > 0:
        period = cfg.period
    else:
        period = estimate_period(z[:, 0])

    if cfg.project_periodic:
        eigvals_used = project_eigs_to_period(eigvals, period=period, radius_eps=cfg.radius_eps)
    else:
        eigvals_used = eigvals.copy()

    rollout_frames = min(cfg.rollout_frames, pos.shape[0] + 2000)
    if args.reconstruct_train:
        rollout_frames = pos.shape[0]
    z_pred = rollout_delay_dmd(
        phi=phi,
        eigvals_proj=eigvals_used,
        y0=y[0],
        z_seed_txr=z,
        delay=cfg.delay,
        rollout_frames=rollout_frames,
    )

    states_pred = z_pred @ pca_basis + mean
    pos_pred = states_pred[:, : n_points * 3].reshape(rollout_frames, n_points, 3)
    vel_pred = states_pred[:, n_points * 3 :].reshape(rollout_frames, n_points, 3)

    target_eval = pos[:rollout_frames]
    vel_target_eval = vel[:rollout_frames]
    n_eval = min(target_eval.shape[0], pos_pred.shape[0])
    center = n_points // 2
    rmse_pos = float(np.sqrt(np.mean((pos_pred[:n_eval] - target_eval[:n_eval]) ** 2)))
    rmse_vel = float(np.sqrt(np.mean((vel_pred[:n_eval] - vel_target_eval[:n_eval]) ** 2)))
    center_loop = float(np.linalg.norm(pos_pred[n_eval - 1, center] - pos_pred[0, center]))
    if n_eval > period:
        cyc = pos_pred[: n_eval - period] - pos_pred[period:n_eval]
        cycle_rmse = float(np.sqrt(np.mean(cyc * cyc)))
    else:
        cycle_rmse = None

    feats = build_point_features(n_points)
    npz_out = os.path.join(args.out_dir, "dmd_rom_dataset.npz")
    np.savez(
        npz_out,
        position=pos_pred.astype(np.float32),
        velocity=vel_pred.astype(np.float32),
        feats=feats.astype(np.float32),
    )

    sequence_files = []
    if args.save_sequence:
        target_gif = os.path.join(args.out_dir, "sequence_target.gif")
        pred_gif = os.path.join(args.out_dir, "sequence_dmd.gif")
        triplet_gif = os.path.join(args.out_dir, "sequence_triplet.gif")
        save_single_gif(
            target_gif,
            target_eval[:n_eval],
            axis=cfg.axis,
            fps=cfg.gif_fps,
            stride=cfg.gif_stride,
            title="target",
            color=(40, 120, 220),
        )
        save_single_gif(
            pred_gif,
            pos_pred[:n_eval],
            axis=cfg.axis,
            fps=cfg.gif_fps,
            stride=cfg.gif_stride,
            title="dmd-rom",
            color=(220, 70, 70),
        )
        save_triplet_gif(
            triplet_gif,
            target_eval[:n_eval],
            pos_pred[:n_eval],
            axis=cfg.axis,
            fps=cfg.gif_fps,
            stride=cfg.gif_stride,
        )
        sequence_files = [target_gif, pred_gif, triplet_gif]

    summary = {
        "dataset": args.dataset,
        "start": start,
        "train_frames": int(pos.shape[0]),
        "n_points": int(n_points),
        "config": cfg.__dict__,
        "period_used": int(period),
        "pca_rank_used": int(z.shape[1]),
        "dmd_modes": int(eigvals.shape[0]),
        "metrics": {
            "trajectory_rmse": rmse_pos,
            "velocity_rmse": rmse_vel,
            "center_loop_closure": center_loop,
            "cycle_rmse_period": cycle_rmse,
        },
        "spectrum": {
            "abs_lambda_mean_before": float(np.mean(np.abs(eigvals))),
            "abs_lambda_mean_after": float(np.mean(np.abs(eigvals_used))),
            "abs_lambda_max_after": float(np.max(np.abs(eigvals_used))),
        },
        "mode": {
            "project_periodic": bool(cfg.project_periodic),
            "reconstruct_train": bool(args.reconstruct_train),
        },
        "outputs": {
            "npz": npz_out,
            "sequence_files": sequence_files,
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done. Outputs:")
    print(f"- {npz_out}")
    for p in sequence_files:
        print(f"- {p}")
    print(f"- {args.out_dir}/summary.json")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
