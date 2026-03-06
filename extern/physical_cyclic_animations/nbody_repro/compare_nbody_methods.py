#!/usr/bin/env python3
"""
Compare a weak baseline and an improved cyclic N-body optimization method.

This script is an approximate visual reproduction inspired by the N-body case
in "Physical Cyclic Animations" (SCA 2023), not an exact paper reimplementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from optimize_nbody_cycle import (
    SimConfig,
    angular_momentum,
    center_of_mass_velocity,
    leapfrog_rollout,
    total_energy,
)


def setup_initial_state(device: torch.device, seed: int, noise_scale: float):
    torch.manual_seed(seed)
    np.random.seed(seed)

    pos0 = torch.tensor(
        [
            [0.97000436, -0.24308753],
            [-0.97000436, 0.24308753],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    mass = torch.ones(3, dtype=torch.float32, device=device)
    vel_base = torch.tensor(
        [
            [0.46620369, 0.43236573],
            [0.46620369, 0.43236573],
            [-0.93240738, -0.86473146],
        ],
        dtype=torch.float32,
        device=device,
    )
    vel_init = vel_base + noise_scale * torch.randn_like(vel_base)
    return pos0, mass, vel_base, vel_init


def run_method(
    name: str,
    cfg: SimConfig,
    pos0: torch.Tensor,
    mass: torch.Tensor,
    vel_init: torch.Tensor,
    vel_base: torch.Tensor,
    iters: int,
    lr: float,
) -> dict:
    vel_param = torch.nn.Parameter(vel_init.clone())
    if name == "weak_gd":
        optimizer = torch.optim.SGD([vel_param], lr=lr)
    elif name == "improved_projected_adam":
        optimizer = torch.optim.Adam([vel_param], lr=lr)
    else:
        raise ValueError(f"Unknown method: {name}")

    history = []

    with torch.no_grad():
        e0_ref = total_energy(pos0, vel_base, mass, cfg.gravity, cfg.softening)
        l0_ref = angular_momentum(pos0, vel_base, mass)

    for i in range(1, iters + 1):
        optimizer.zero_grad()

        if name == "weak_gd":
            # Weak method: optimize only positional closure, no projection.
            v0 = vel_param
            traj_pos, traj_vel = leapfrog_rollout(pos0, v0, mass, cfg)
            closure_pos = ((traj_pos[-1] - pos0) ** 2).mean()
            closure_vel = ((traj_vel[-1] - v0) ** 2).mean()
            loss = closure_pos
        else:
            # Improved method: COM projection + closure + conservation regularizers.
            v0 = vel_param - center_of_mass_velocity(vel_param, mass)[None, :]
            traj_pos, traj_vel = leapfrog_rollout(pos0, v0, mass, cfg)
            closure_pos = ((traj_pos[-1] - pos0) ** 2).mean()
            closure_vel = ((traj_vel[-1] - v0) ** 2).mean()

            e0 = total_energy(pos0, v0, mass, cfg.gravity, cfg.softening)
            et = total_energy(traj_pos[-1], traj_vel[-1], mass, cfg.gravity, cfg.softening)
            energy_drift = ((et - e0) / (torch.abs(e0_ref) + 1e-8)) ** 2

            l0 = angular_momentum(pos0, v0, mass)
            lt = angular_momentum(traj_pos[-1], traj_vel[-1], mass)
            angular_drift = ((lt - l0) / (torch.abs(l0_ref) + 1e-8) if torch.abs(l0_ref) > 1e-8 else (lt - l0)) ** 2

            loss = closure_pos + 0.3 * closure_vel + 0.05 * energy_drift + 0.05 * angular_drift

        loss.backward()
        optimizer.step()

        history.append(
            {
                "iter": i,
                "loss": float(loss.item()),
                "closure_pos": float(closure_pos.item()),
                "closure_vel": float(closure_vel.item()),
            }
        )

    with torch.no_grad():
        if name == "weak_gd":
            v0_final = vel_param
        else:
            v0_final = vel_param - center_of_mass_velocity(vel_param, mass)[None, :]
        traj_pos, traj_vel = leapfrog_rollout(pos0, v0_final, mass, cfg)
        pos_t = traj_pos[-1]
        vel_t = traj_vel[-1]
        metrics = {
            "closure_pos_rmse": float(torch.sqrt(((pos_t - pos0) ** 2).mean()).item()),
            "closure_vel_rmse": float(torch.sqrt(((vel_t - v0_final) ** 2).mean()).item()),
            "energy_start": float(total_energy(pos0, v0_final, mass, cfg.gravity, cfg.softening).item()),
            "energy_end": float(total_energy(pos_t, vel_t, mass, cfg.gravity, cfg.softening).item()),
            "angular_start": float(angular_momentum(pos0, v0_final, mass).item()),
            "angular_end": float(angular_momentum(pos_t, vel_t, mass).item()),
        }

    return {
        "name": name,
        "history": history,
        "traj_pos": traj_pos.detach().cpu().numpy(),
        "traj_vel": traj_vel.detach().cpu().numpy(),
        "metrics": metrics,
    }


def save_history_csv(path: str, history: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "loss", "closure_pos", "closure_vel"])
        writer.writeheader()
        writer.writerows(history)


def plot_loss_curve(weak: dict, improved: dict, out_path: str):
    plt.figure(figsize=(7.5, 4.8))
    xw = [d["iter"] for d in weak["history"]]
    yw = [d["closure_pos"] for d in weak["history"]]
    xi = [d["iter"] for d in improved["history"]]
    yi = [d["closure_pos"] for d in improved["history"]]

    plt.semilogy(xw, yw, label="Weak baseline (GD, closure only)", linewidth=2.0)
    plt.semilogy(xi, yi, label="Improved (projected + regularized)", linewidth=2.2)
    plt.xlabel("Optimization Iteration")
    plt.ylabel("Position Closure MSE (log scale)")
    plt.title("N-body Cyclic Optimization: Convergence Comparison")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_trajectories(weak: dict, improved: dict, out_path: str):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))

    for ax, res, title in [
        (axes[0], weak, "Weak baseline"),
        (axes[1], improved, "Improved method"),
    ]:
        traj = res["traj_pos"]  # (T, 3, 2)
        for b in range(3):
            ax.plot(traj[:, b, 0], traj[:, b, 1], color=colors[b], linewidth=1.6, label=f"Body {b+1}")
            ax.scatter(traj[0, b, 0], traj[0, b, 1], color=colors[b], s=28, marker="o")
            ax.scatter(traj[-1, b, 0], traj[-1, b, 1], color=colors[b], s=28, marker="x")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("N-body Trajectory over One Period (start: circle, end: cross)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_panel_figure(loss_png: str, traj_png: str, out_png: str):
    left = Image.open(loss_png).convert("RGB")
    right = Image.open(traj_png).convert("RGB")
    h = max(left.height, right.height)
    left = ImageOps.pad(left, (left.width, h), color="white")
    right = ImageOps.pad(right, (right.width, h), color="white")

    pad = 30
    canvas = Image.new("RGB", (left.width + right.width + 3 * pad, h + 2 * pad), "white")
    canvas.paste(left, (pad, pad))
    canvas.paste(right, (left.width + 2 * pad, pad))

    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), "A. Convergence Comparison", fill="black")
    draw.text((left.width + 2 * pad, 8), "B. Trajectory Closure Comparison", fill="black")
    canvas.save(out_png)


def parse_args():
    p = argparse.ArgumentParser(description="Compare weak and improved N-body cyclic methods.")
    p.add_argument("--iters", type=int, default=1200)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.003)
    p.add_argument("--gravity", type=float, default=1.0)
    p.add_argument("--softening", type=float, default=1e-2)
    p.add_argument("--noise_scale", type=float, default=0.18)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lr_weak", type=float, default=0.02)
    p.add_argument("--lr_improved", type=float, default=0.03)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out_dir", type=str, default="outputs/compare_paper_style")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    cfg = SimConfig(steps=args.steps, dt=args.dt, gravity=args.gravity, softening=args.softening)
    os.makedirs(args.out_dir, exist_ok=True)

    pos0, mass, vel_base, vel_init = setup_initial_state(device, args.seed, args.noise_scale)

    weak = run_method(
        "weak_gd",
        cfg=cfg,
        pos0=pos0,
        mass=mass,
        vel_init=vel_init,
        vel_base=vel_base,
        iters=args.iters,
        lr=args.lr_weak,
    )
    improved = run_method(
        "improved_projected_adam",
        cfg=cfg,
        pos0=pos0,
        mass=mass,
        vel_init=vel_init,
        vel_base=vel_base,
        iters=args.iters,
        lr=args.lr_improved,
    )

    save_history_csv(os.path.join(args.out_dir, "weak_history.csv"), weak["history"])
    save_history_csv(os.path.join(args.out_dir, "improved_history.csv"), improved["history"])
    np.save(os.path.join(args.out_dir, "weak_traj.npy"), weak["traj_pos"])
    np.save(os.path.join(args.out_dir, "improved_traj.npy"), improved["traj_pos"])

    summary = {
        "config": asdict(cfg),
        "seed": args.seed,
        "weak": weak["metrics"],
        "improved": improved["metrics"],
    }
    with open(os.path.join(args.out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    loss_png = os.path.join(args.out_dir, "figure_loss_compare.png")
    traj_png = os.path.join(args.out_dir, "figure_trajectory_compare.png")
    panel_png = os.path.join(args.out_dir, "figure_paper_style_panel.png")
    plot_loss_curve(weak, improved, loss_png)
    plot_trajectories(weak, improved, traj_png)
    make_panel_figure(loss_png, traj_png, panel_png)

    print("Done. Outputs:")
    print(f"- {loss_png}")
    print(f"- {traj_png}")
    print(f"- {panel_png}")
    print(f"- {args.out_dir}/summary_metrics.json")
    print("\nSummary metrics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
