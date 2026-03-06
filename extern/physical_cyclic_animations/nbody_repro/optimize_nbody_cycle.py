#!/usr/bin/env python3
"""
Simplified N-body cyclic optimization demo inspired by:
Jia et al., Physical Cyclic Animations (SCA 2023), DOI: 10.1145/3606938
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SimConfig:
    steps: int
    dt: float
    gravity: float
    softening: float


def pairwise_acceleration(pos: torch.Tensor, mass: torch.Tensor, gravity: float, softening: float) -> torch.Tensor:
    """
    pos: (N, 2), mass: (N,)
    returns acceleration (N, 2)
    """
    diff = pos[None, :, :] - pos[:, None, :]  # (N, N, 2), x_j - x_i
    dist2 = (diff * diff).sum(dim=-1) + softening * softening  # (N, N)

    # Avoid self-force by zeroing diagonal contributions.
    inv_dist3 = torch.where(
        torch.eye(pos.shape[0], device=pos.device, dtype=torch.bool),
        torch.zeros_like(dist2),
        dist2.pow(-1.5),
    )
    # a_i = G * sum_j m_j * (x_j - x_i) / |x_j - x_i|^3
    acc = gravity * (diff * inv_dist3[..., None] * mass[None, :, None]).sum(dim=1)
    return acc


def leapfrog_rollout(
    pos0: torch.Tensor, vel0: torch.Tensor, mass: torch.Tensor, cfg: SimConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      positions: (steps+1, N, 2)
      velocities: (steps+1, N, 2)
    """
    pos = pos0
    vel = vel0
    positions = [pos]
    velocities = [vel]

    acc = pairwise_acceleration(pos, mass, cfg.gravity, cfg.softening)
    vel_half = vel + 0.5 * cfg.dt * acc

    for _ in range(cfg.steps):
        pos = pos + cfg.dt * vel_half
        acc_new = pairwise_acceleration(pos, mass, cfg.gravity, cfg.softening)
        vel = vel_half + 0.5 * cfg.dt * acc_new
        vel_half = vel_half + cfg.dt * acc_new

        positions.append(pos)
        velocities.append(vel)

    return torch.stack(positions, dim=0), torch.stack(velocities, dim=0)


def total_energy(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor, gravity: float, softening: float) -> torch.Tensor:
    kin = 0.5 * (mass[:, None] * vel * vel).sum()
    diff = pos[None, :, :] - pos[:, None, :]
    dist = torch.sqrt((diff * diff).sum(dim=-1) + softening * softening)
    i, j = torch.triu_indices(pos.shape[0], pos.shape[0], offset=1)
    pot = -(gravity * mass[i] * mass[j] / dist[i, j]).sum()
    return kin + pot


def angular_momentum(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    # 2D scalar z-component: x*vy - y*vx
    cross = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]
    return (mass * cross).sum()


def center_of_mass_velocity(vel: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    return (mass[:, None] * vel).sum(dim=0) / mass.sum()


def optimize_cycle(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    cfg = SimConfig(
        steps=args.steps,
        dt=args.dt,
        gravity=args.gravity,
        softening=args.softening,
    )

    # Figure-eight inspired 3-body initial positions.
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

    # Initialize velocities from a known periodic pattern with perturbation.
    vel_base = torch.tensor(
        [
            [0.46620369, 0.43236573],
            [0.46620369, 0.43236573],
            [-0.93240738, -0.86473146],
        ],
        dtype=torch.float32,
        device=device,
    )
    vel_init = vel_base + args.noise_scale * torch.randn_like(vel_base)
    vel_param = torch.nn.Parameter(vel_init.clone())
    optimizer = torch.optim.Adam([vel_param], lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    loss_history: list[tuple[int, float, float, float]] = []

    # Static references
    with torch.no_grad():
        e0_ref = total_energy(pos0, vel_base, mass, cfg.gravity, cfg.softening)
        l0_ref = angular_momentum(pos0, vel_base, mass)

    for it in range(1, args.iters + 1):
        optimizer.zero_grad()

        # Encourage zero COM velocity each iteration.
        v0 = vel_param - center_of_mass_velocity(vel_param, mass)[None, :]
        traj_pos, traj_vel = leapfrog_rollout(pos0, v0, mass, cfg)
        pos_t = traj_pos[-1]
        vel_t = traj_vel[-1]

        closure_pos = ((pos_t - pos0) ** 2).mean()
        closure_vel = ((vel_t - v0) ** 2).mean()

        e0 = total_energy(pos0, v0, mass, cfg.gravity, cfg.softening)
        et = total_energy(pos_t, vel_t, mass, cfg.gravity, cfg.softening)
        energy_drift = ((et - e0) / (torch.abs(e0_ref) + 1e-8)) ** 2

        l0 = angular_momentum(pos0, v0, mass)
        lt = angular_momentum(pos_t, vel_t, mass)
        angular_drift = ((lt - l0) / (torch.abs(l0_ref) + 1e-8) if torch.abs(l0_ref) > 1e-8 else (lt - l0)) ** 2

        loss = (
            args.w_closure_pos * closure_pos
            + args.w_closure_vel * closure_vel
            + args.w_energy * energy_drift
            + args.w_angular * angular_drift
        )
        loss.backward()
        optimizer.step()

        loss_history.append((it, float(loss.item()), float(closure_pos.item()), float(closure_vel.item())))
        if it % args.log_every == 0 or it == 1 or it == args.iters:
            print(
                f"[{it:4d}/{args.iters}] "
                f"loss={loss.item():.6e} "
                f"closure_pos={closure_pos.item():.6e} "
                f"closure_vel={closure_vel.item():.6e}"
            )

    with torch.no_grad():
        v0_opt = vel_param - center_of_mass_velocity(vel_param, mass)[None, :]
        traj_pos, traj_vel = leapfrog_rollout(pos0, v0_opt, mass, cfg)
        pos_t = traj_pos[-1]
        vel_t = traj_vel[-1]

        metrics = {
            "steps": cfg.steps,
            "dt": cfg.dt,
            "period": cfg.steps * cfg.dt,
            "closure_pos_rmse": float(torch.sqrt(((pos_t - pos0) ** 2).mean()).item()),
            "closure_vel_rmse": float(torch.sqrt(((vel_t - v0_opt) ** 2).mean()).item()),
            "energy_start": float(total_energy(pos0, v0_opt, mass, cfg.gravity, cfg.softening).item()),
            "energy_end": float(total_energy(pos_t, vel_t, mass, cfg.gravity, cfg.softening).item()),
            "angular_start": float(angular_momentum(pos0, v0_opt, mass).item()),
            "angular_end": float(angular_momentum(pos_t, vel_t, mass).item()),
        }

    np.save(os.path.join(args.out_dir, "trajectory.npy"), traj_pos.detach().cpu().numpy())
    np.save(os.path.join(args.out_dir, "velocities.npy"), traj_vel.detach().cpu().numpy())

    with open(os.path.join(args.out_dir, "loss_history.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "loss", "closure_pos", "closure_vel"])
        writer.writerows(loss_history)

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nFinal metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote outputs to: {args.out_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified N-body cyclic optimization demo.")
    parser.add_argument("--iters", type=int, default=1200)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--softening", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--noise_scale", type=float, default=0.15)
    parser.add_argument("--w_closure_pos", type=float, default=1.0)
    parser.add_argument("--w_closure_vel", type=float, default=0.3)
    parser.add_argument("--w_energy", type=float, default=0.05)
    parser.add_argument("--w_angular", type=float, default=0.05)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out_dir", type=str, default="outputs/default_run")
    return parser.parse_args()


if __name__ == "__main__":
    optimize_cycle(parse_args())
