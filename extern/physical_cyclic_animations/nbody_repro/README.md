# N-Body Cyclic Reproduction (Simplified)

This folder contains a lightweight, differentiable N-body cyclic optimization demo inspired by:

- Jia et al., *Physical Cyclic Animations* (SCA 2023)  
  DOI: https://doi.org/10.1145/3606938

## What this demo does

- Simulates a 2D gravitational 3-body system with a differentiable leapfrog integrator.
- Optimizes initial velocities so the trajectory approximately closes after one period.
- Writes trajectory and optimization logs for inspection.

This is **not** a full reimplementation of the paper's solver.  
It is a compact reproduction-style experiment for the N-body case.

## Run

```bash
cd /workspace/cyclic_animation/extern/physical_cyclic_animations/nbody_repro
python3 optimize_nbody_cycle.py --iters 1200 --steps 1000 --dt 0.003 --out_dir outputs/default_run
```

## Outputs

- `metrics.json`: closure and conservation metrics
- `trajectory.npy`: shape `(steps+1, 3, 2)` positions
- `velocities.npy`: shape `(steps+1, 3, 2)` velocities
- `loss_history.csv`: optimization curve

## Key arguments

- `--iters`: optimization iterations
- `--steps`: simulation steps in one cycle
- `--dt`: simulation timestep
- `--lr`: optimizer learning rate
- `--noise_scale`: perturbation added to initial velocities
- `--out_dir`: output directory

## Weak-vs-Improved Comparison Figures

To generate side-by-side comparison plots (paper-style visual check):

```bash
cd /workspace/cyclic_animation/extern/physical_cyclic_animations/nbody_repro
python3 compare_nbody_methods.py --iters 1200 --steps 1000 --dt 0.003 --out_dir outputs/compare_paper_style
```

This writes:

- `figure_loss_compare.png` (convergence curve)
- `figure_trajectory_compare.png` (trajectory closure comparison)
- `figure_paper_style_panel.png` (single combined panel for quick inspection)
- `summary_metrics.json`
