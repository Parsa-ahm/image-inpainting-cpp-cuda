# image-inpainting-cpp-cuda — Project Charter

**Status:** ACTIVE — Rung 0 in progress (build + GPU sanity green; data pipeline / PRNG pending).
This document is the single source of truth for *why* this project exists and *what done means*.
If work ever drifts from this, either the work is wrong or this doc is wrong — reconcile before
continuing. Do not lose the plot.

> Name: `image-inpainting-cpp-cuda`. Locked 2026-08-25.

---

## 0. The one sentence

**Write, by hand, the full GPU training of a legit modern deep-learning model (a score-based
generative model of faces) — every kernel, forward and backward — to understand exactly how the
bits move on the GPU, and make it fast enough to stand next to cuBLAS/cuDNN.**

## 1. One-line summary

From first principles in C++/CUDA with zero ML frameworks: a **noise-conditional score network**
trained **entirely on the GPU with hand-written kernels** (forward + backward), sampled by
hand-written CUDA Langevin kernels, and demonstrated on **face inpainting** (CelebA — erase a
region, the model reconstructs it). The whole point is to own the GPU training path end to end.

## 2. Why this project exists (the purpose)

Career purpose: **capstone of the straight-ML phase.** After this, Parsa moves to kernels or
agents. Its job is to **erase any reasonable doubt that Parsa is a real ML engineer** by proving
three things most candidates cannot prove together:

1. **Depth** — understands generative modeling from the score/energy up, not via `.fit()`.
2. **GPU systems** — has *personally written and optimized* the CUDA kernels for the entire
   training and sampling path of a modern model. Knows how every bit moves on the GPU.
3. **Frontier literacy** — places the work correctly in the current field (score-based models,
   diffusion, flow matching, consistency models) and talks about it fluently.

Extends the two existing pillars:
- **pml-notes** (from-scratch Murphy *Probabilistic ML*) — the *generative* capstone of that book.
- **ml-jax-pytorch** (efficiency / systems-ML) — this is the from-scratch GPU-kernels escalation.

## 3. Thesis (the defensible claim)

> **Claim:** "I built a modern deep generative model — a score network trained by denoising score
> matching and sampled with annealed Langevin — from scratch in C++/CUDA with no ML framework,
> including hand-writing and optimizing every GPU kernel for **both forward and backward passes and
> the full training loop**. I benchmarked my kernels against cuBLAS/cuDNN and can explain, defend,
> and profile every line."

About **demonstrated capability + understanding + a working, optimized artifact.** Does NOT depend
on beating a published quality benchmark. Lesson banked from the MoE project: **do not stake the
project on a comparative thesis you cannot cleanly support.**

## 4. Non-goals — what we explicitly do NOT claim

- **NOT** state-of-the-art image quality. Small faces, modest model, from scratch. Samples should be
  *recognizable faces*, not photorealistic.
- **NOT** beating cuBLAS/cuDNN. We *benchmark against* them and close the gap as far as possible;
  matching NVIDIA's assembly-tuned kernels exactly is a research effort, not the bar (see §10, §13).
- **NOT** a general ML framework. One model family, one demo, done well.
- Earlier draft framed this around thermodynamic computing / Extropic — **deliberately dropped**
  (read as an Extropic job application; invited defending a field outside our expertise). No
  thermodynamic/p-bit/quantum framing anywhere in the repo.

## 5. Objectives (concrete, measurable)

- **O1.** A **hand-written counter-based PRNG** (Philox-style), identical on CPU and across parallel
  GPU threads.
- **O2.** A **GPU tensor + core-kernel library**: device memory, elementwise ops, and an **optimized
  tiled SGEMM (matmul)** — shared memory, register blocking, coalesced/vectorized access —
  benchmarked against cuBLAS.
- **O3.** A **GPU reverse-mode autodiff**: forward + **backward** kernels for matmul, conv,
  activations, and normalization, gradient-checked against a CPU finite-difference oracle.
- **O4.** A **noise-conditional score network** (small CNN) — inputs image + noise level σ, outputs
  the score — with the **entire training loop on the GPU** (DSM loss + optimizer, all hand-written).
- **O5.** **Annealed Langevin sampling** in CUDA producing recognizable face samples.
- **O6.** The **face inpainting demo**: erase a region → reconstruct via conditional sampling.
- **O7.** **Optimization + metrics + writeup**: per-kernel benchmarks vs cuBLAS/cuDNN with a roofline
  analysis, the full metric suite, and a writeup deriving the math and situating the work.

## 6. The ML spine (RESOLVED: continuous / score-based — Path B)

> **Score-based generative models ⟺ energy-based models ⟺ MCMC sampling.**

- The model learns a **score** = ∇ₓ log p(x) = **−∇ₓ E(x)**.
- Trained by **denoising score matching** (Vincent 2011): noise x → x̃ = x + σε, regress the
  network's score output onto the analytic target **(x − x̃)/σ² = −ε/σ**. Stable, no
  partition-function estimation. (Equivalent up to scaling to DDPM ε-prediction — we output score.)
- The score net is **noise-conditional** (takes σ) — required for annealed sampling over σ levels.
- Sampling = **annealed Langevin dynamics** (Song & Ermon 2019): from noise, step along the score
  plus a little noise, annealing σ down. *Unadjusted* Langevin (ULA), no Metropolis accept/reject.
- **Inpainting = conditional sampling:** run Langevin on the erased region while clamping observed
  pixels each step.

Concepts owned by the end: score matching & DSM · Langevin (ULA vs MALA, detailed balance) ·
forward/reverse diffusion & noise schedules (DDPM) · score-based SDEs · EBM ↔ score equivalence ·
and *situated* knowledge of flow matching / rectified flow and consistency models. Plus the GPU
side: memory coalescing, shared-memory tiling, register blocking, occupancy, kernel fusion,
roofline/arithmetic intensity.

## 7. The path (build ladder — each rung is shippable)

Everything the model does runs on the **GPU** and is **written by Parsa**. CPU code exists only as
Claude's test oracle (see §8). MNIST-grayscale is a fast *correctness rig* for early rungs; **CelebA
32×32 (grayscale → RGB, then 64×64 if time) is the real target.**

- **Rung 0 — Scaffold.** CMake+CUDA build (✓ builds, reports the GPU); CelebA download+resize
  (Claude's Python data-prep) → simple binary blob; C++ loader; hand-written Philox PRNG.
  *Artifact:* builds; loads and re-saves a face image; PRNG passes KAT + statistical checks.
- **Rung 1 — GPU core kernels.** Device tensor, elementwise ops, and the **optimized tiled SGEMM**.
  *Artifact:* matmul matches the CPU oracle; benchmark vs cuBLAS (GFLOP/s, % of cuBLAS). **← systems
  floor; a real standalone flex on its own.**
- **Rung 2 — GPU autodiff.** Forward + backward kernels for the ops the model needs (matmul, conv,
  activations, norm). *Artifact:* gradient check passes against CPU finite-difference oracle.
- **Rung 3 — GPU training.** Noise-conditional CNN + DSM loss + optimizer, full loop on GPU.
  Validate on MNIST-gray (fast), then train CelebA-32 gray. *Artifact:* loss descends; recognizable
  face samples. **← ML floor; quality risk starts here.**
- **Rung 4 — Sampling.** Annealed Langevin sampling kernels → unconditional face generation.
  *Artifact:* a grid of generated faces.
- **Rung 5 — Inpainting.** Conditional sampling with clamped observed pixels; upgrade to RGB / 64×64
  if time. *Artifact:* the before/after face-reconstruction demo. **← the wow.**
- **Rung 6 — Optimize + metrics + writeup.** Profile and push kernels toward library-competitive;
  roofline; FID/PSNR/SSIM; benchmark vs cuBLAS/cuDNN; the writeup. *Artifact:* a repo a senior eng or
  researcher would accept.

## 8. Collaboration model (STRICT — this is how we work)

**Parsa writes 100% of the implementation.** Every line of C++/CUDA that is the model, the kernels,
the autodiff, the training loop, and the sampler is his — he writes the **GPU** code, not CPU code.
The whole point is that he can produce it himself and therefore understands it.

**Claude does NOT show implementation code for Parsa's parts.** Claude's job:
- **Explain the concept** and the math deeply enough that Parsa can derive the implementation.
- **Define precise task specs**: inputs/outputs, shapes, invariants, numerical targets, edge cases —
  *what* the kernel/function must do and how to reason about it, never *the code that does it*.
- **Write the tests** and a **CPU reference oracle** inside the tests (the oracle validates his GPU
  kernels; naive CPU reference math is fine — it never reveals the optimized GPU technique).
- Handle **build config, data-prep scripts, plotting/benchmark harnesses, reviews**.
- **Optimize together:** once Parsa has a correct kernel, Claude and Parsa profile and tune it as a
  pair — here Claude may discuss specific techniques and diffs, because now it's optimization of
  *his* working code, not handing him the solution.

Performance bar: **fully optimized — apply every standard technique and benchmark against
cuBLAS/cuDNN.** Realistic target: within a documented factor of the libraries (stretch: match).

## 9. Boundaries (from-scratch line — decided, do not relitigate)

- **Core = from scratch, no libraries:** PRNG, tensor math, all GPU kernels (fwd+bwd), autodiff,
  training loop, optimizer, Langevin sampling.
- **Allowed thin non-ML utilities:** CMake; single-header image writer or raw PPM; the C++ standard
  library; CUDA runtime (launch/memory) — **not cuBLAS/cuDNN** (except as a *benchmark reference* to
  measure against, never called in the model path).
- **cuRAND** not used — PRNG is hand-written. **cuBLAS/cuDNN** appear only in the benchmark harness.
- **Data-prep + plotting** may be thin Python (resize CelebA, render charts/GIFs). Not ML, not part
  of the from-scratch claim.
- **Evaluation-only exception:** one external **pretrained** Inception, solely to compute **FID**.
  Never touches model/training/sampling.

**Stack:** C++17 host + hand-written CUDA. Dev box: **RTX 2070 SUPER (`sm_75`, 8 GB, 40 SMs)**, CUDA
12.0 (`nvcc`), CMake ≥ 3.24, host compiler **g++-12** (CUDA 12.0 rejects g++-13).

## 10. Metrics (track everything)

- **Kernel performance (central):** GFLOP/s and **% of cuBLAS/cuDNN** per kernel; roofline /
  arithmetic-intensity plot; occupancy; before/after each optimization.
- **Training:** loss curves, gradient norms, per-layer stats, step time / throughput.
- **Sampler health:** score magnitude over σ levels, autocorrelation, mixing.
- **Inpainting fidelity vs ground truth:** PSNR, SSIM, MSE (hand-coded).
- **Sample realism:** our own small classifier scoring samples; **FID** via the eval-only Inception.
- **Eyeball:** face sample grids and before/after inpainting panels.

## 11. Showcase pieces

- **Face inpainting** — primary wow ("erased half the face, the model rebuilt it").
- **Generated-face grid** — the model dreaming up faces.
- **Kernel-vs-cuBLAS performance charts + roofline** — the credibility layer proving the GPU work is
  real and optimized, not a slow toy.

## 12. Success criteria (definition of done)

- All of O1–O7 satisfied.
- Every rung's artifact reproducible from a clean clone with documented commands.
- Parsa can, cold, explain and profile any kernel, derive the §6 math, and place the work in the
  field.
- Kernels benchmarked vs cuBLAS/cuDNN with the gap measured and explained (not necessarily closed).
- Writeup passes: "a senior ML engineer or researcher would accept this as real."
- **Timebox: aim ~4–5 weeks to a strong Rung 5; the RGB/64×64 + full optimization pass may run
  longer.** If time is short, ship at Rung 3 (trained + recognizable faces) with honest future work —
  never ship a broken higher rung.

## 13. Risks and mitigations

- **R1 — Scope/timebox.** GPU training from scratch, written solo while learning, is big.
  *Mitigation:* rung ladder with shippable floors (Rung 1 systems flex, Rung 3 ML floor); MNIST rig
  before CelebA; grayscale before RGB.
- **R2 — Sample quality.** *Mitigation:* DSM is stable; faces are a smooth single-domain manifold
  (more learnable than CIFAR at small scale); lead with inpainting (constrained → looks good easier).
- **R3 — Optimization bar is a trap if taken as "equal cuBLAS."** *Mitigation:* §10 treats it as a
  measured goal with documented gap, not a pass/fail gate. Within ~1.5–2× is excellent.
- **R4 — GPU-autodiff/backward-kernel difficulty.** Backprop kernels are the hardest part.
  *Mitigation:* rigorous gradient checks vs CPU oracle at every op before composing them.
- **R5 — Losing the plot (the MoE failure mode).** *Mitigation:* this charter; re-read §3/§4 before
  any writeup.

## 14. References

- Hyvärinen 2005 — Score Matching. · Vincent 2011 — Denoising Score Matching. · Welling & Teh 2011 —
  SGLD. · Song & Ermon 2019; Song et al. 2021 — score-based modeling / annealed Langevin / SDEs. ·
  Ho, Jain, Abbeel 2020 — DDPM (situating). · Lipman 2022; Liu 2022 — Flow Matching / Rectified Flow
  (situating). · Song 2023 — Consistency Models (situating). · Salmon et al. 2011 — Philox PRNG. ·
  Murphy — *PML: Advanced Topics* (pml-notes book): EBMs, MCMC, diffusion.
- GPU: NVIDIA CUDA C++ Programming/Best-Practices guides; a tiled-SGEMM optimization walkthrough as
  the perf reference target.

## 15. What Parsa will be able to say afterward (the payoff)

- "I hand-wrote and optimized every CUDA kernel — forward *and* backward — for the full training of a
  modern score-based generative model. No framework touched the core."
- "I benchmarked my SGEMM and conv against cuBLAS/cuDNN and can show you the roofline and where the
  gap is and why."
- "My face-inpainting demo is conditional annealed-Langevin sampling — here's how it relates to DDPM
  and why frontier labs moved to flow matching."
