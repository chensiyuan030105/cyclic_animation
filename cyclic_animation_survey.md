# Cyclic Animation Survey for a Koopman-Based Project

## 1. Scope and Motivation

This note surveys cyclic character animation from a Computer Graphics perspective, with an emphasis on SCA/SIGGRAPH/TOG/Eurographics practice.

Cyclic animation methods are typically judged on four criteria:

- visual plausibility of repeated motion,
- controllability (speed, heading, style, transitions),
- contact correctness (especially foot contact),
- and long-horizon stability (no phase drift over many cycles).

## 2. Historical Trajectory in the CG Community

- **2002 — Motion Graphs (SIGGRAPH)**  
  Built cyclic and transitionable motion from clip graphs and transition edges.
- **2010 — Motion Fields (TOG)**  
  Shifted from discrete clip traversal to continuous control-oriented motion representation.
- **2016 — Deep Learning Framework for Character Motion (TOG/SIGGRAPH Asia)**  
  Established modern deep data-driven pipelines for synthesis and editing.
- **2017 — PFNN (TOG)**  
  Introduced explicit phase conditioning, becoming a key baseline for locomotion control.
- **2020 — Local Motion Phases (TOG)**  
  Extended single global phase to local/contact-aware phase modeling.
- **2022 — DeepPhase (TOG)**  
  Learned periodic latent manifolds with less manual phase engineering.
- **2023+ — Phase-Manifold Transition Methods**  
  Used manifold structure for in-betweening and robust transitions, with extensions to new morphologies.

## 3. Core Technical Challenges

- **Phase consistency:** preserving periodic structure under autoregressive rollout.
- **Contact fidelity:** preventing foot sliding and contact timing errors.
- **Transition quality:** handling cross-speed, cross-style, and cross-action switching.
- **Long-horizon robustness:** avoiding accumulation of small errors over many cycles.

## 4. Dominant Methodological Pattern

A recurring CG pattern is:

- phase-aware representation (explicit phase signal or learned periodic manifold),
- structured temporal model (often conditioned or mixture-based),
- contact-aware losses and constraints,
- and evaluation beyond short reconstruction windows.

## 5. Why Koopman Is a Strong Fit

For cyclic motion, Koopman-style modeling offers a principled way to impose stable latent dynamics:

\[
z_{t+1} = K z_t + B u_t
\]

For a CG-native cyclic animation pipeline, the practical design is:

- learn dynamics in a **phase-aware latent space** rather than raw joint space,
- regularize the spectrum of \(K\) near the unit circle for periodic stability,
- enforce **cycle consistency** (\(x_{t+P} \approx x_t\)),
- enforce **contact/physics losses** to keep motion physically plausible,
- and optionally add a nonlinear residual branch for expressive detail.

## 6. Recommended Evaluation Protocol (SCA-Style)

- **Datasets:** AMASS locomotion subsets, optional style-rich sets (e.g., 100STYLE).
- **Metrics:** long-horizon rollout error, foot-skating/contact violations, phase drift in frequency space.
- **Ablations:** remove spectral constraints, remove cycle loss, remove contact loss, vary latent dimension.
- **Presentation:** include long-rollout videos and difficult control transitions.

## 7. Suggested Paper Positioning for SCA 2026

Use this narrative order:

1. Cyclic locomotion remains unstable in long-horizon neural rollout.
2. A Koopman phase-latent formulation provides stable and controllable cyclic dynamics.
3. Contact-aware constraints are necessary to translate latent stability into visual realism.
4. The combined method improves long-horizon quality versus phase-based autoregressive baselines.

## 8. Curated References

- Motion Graphs (2002): https://graphics.cs.wisc.edu/Papers/2002/KGP02/mograph.pdf
- Motion Fields (2010): https://homes.cs.washington.edu/~jovan/papers/lee-2010-mf.pdf
- Deep Learning Framework (2016): https://doi.org/10.1145/2897824.2925975
- PFNN (2017): https://www.pure.ed.ac.uk/ws/files/35467734/phasefunction.pdf
- Local Motion Phases (2020): https://doi.org/10.1145/3386569.3392450
- DeepPhase (2022): https://doi.org/10.1145/3528223.3530178
- Motion In-Betweening with Phase Manifolds (2023): https://arxiv.org/abs/2308.12751
- WalkTheDog (2024): https://arxiv.org/abs/2407.18946
- AMASS dataset: https://arxiv.org/abs/1904.03278
- 100STYLE dataset: https://www.ianxmason.com/100style/

## 9. Additional CG References (Including Albert Chern)

### 9.1 Cyclic Animation and Phase-Based Motion in CG

- Jia, Wang, Li, Chern. *Physical Cyclic Animations* (PACM CGIT / SCA 2023): https://doi.org/10.1145/3606938
- Holden, Saito, Komura. *A Deep Learning Framework for Character Motion Synthesis and Editing* (TOG 2016): https://doi.org/10.1145/2897824.2925975
- Holden, Komura, Saito. *Phase-Functioned Neural Networks for Character Control* (TOG 2017): https://doi.org/10.1145/3072959.3073663
- Starke et al. *Local Motion Phases for Learning Multi-Contact Character Movements* (TOG 2020): https://doi.org/10.1145/3386569.3392450
- Starke, Mason, Komura. *DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds* (TOG 2022): https://doi.org/10.1145/3528223.3530178

### 9.2 Albert Chern: Geometry, Fluids, and Physics-Based Graphics

- Chern et al. *Schrödinger's Smoke* (TOG / SIGGRAPH 2016): https://doi.org/10.1145/2897824.2925868
- Chern et al. *Inside Fluids: Clebsch Maps for Visualization and Processing* (TOG / SIGGRAPH 2017): https://doi.org/10.1145/3072959.3073591
- Chern et al. *Shape from Metric* (TOG / SIGGRAPH 2018): https://doi.org/10.1145/3197517.3201276
- Nabizadeh et al. *Covector Fluids* (TOG / SIGGRAPH 2022): https://doi.org/10.1145/3528223.3530120
- Ishida, Wojtan, Chern. *Hidden Degrees of Freedom in Implicit Vortex Filaments* (TOG / SIGGRAPH Asia 2022): https://doi.org/10.1145/3550454.3555459
- Nabizadeh et al. *Fluid Implicit Particles on Coadjoint Orbits* (TOG / SIGGRAPH Asia 2024): https://doi.org/10.1145/3687970

### 9.3 Author/Project Index

- Albert Chern project/publication list: https://cseweb.ucsd.edu/~alchern/projects/
- Physical Cyclic Animations project page (code/video/bib): https://shiyang-jia.com/physical_cyclic_animations/
