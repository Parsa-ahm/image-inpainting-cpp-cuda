# image-inpainting-cpp-cuda — Project Charter

**Status:** charter draft, pre-code. This document is the single source of truth for *why* this
project exists and *what done means*. If work ever drifts from this, either the work is wrong or
this doc is wrong — reconcile before continuing. Do not lose the plot.

> Working name: `image-inpainting-cpp-cuda` (what it does). Not locked.

---

## 1. One-line summary

Build, from first principles in C++/CUDA with zero ML frameworks, a **score-based / energy-based
generative image model** — hand-written autodiff, a trained score network, and hand-written CUDA
sampling kernels — and demonstrate it on **image inpainting** (image with a region removed →
model fills it back in).

## 2. Why this project exists (the purpose)

Career purpose: **capstone of the straight-ML phase.** After this, Parsa moves to kernels or
agents. Its job is to **erase any reasonable doubt that Parsa is a real ML engineer** by proving
three things most candidates cannot prove together:

1. **Depth** — understands generative modeling from the score/energy up, not via `.fit()`.
2. **Systems** — writes the CUDA kernels that are the performance-critical heart of sampling.
3. **Frontier literacy** — can place the work correctly in the current field (score-based models,
   diffusion, flow matching, consistency models) and talk about it fluently.

Extends the two existing pillars:
- **pml-notes** (from-scratch reimplementation of Murphy's *Probabilistic ML*) — this is the
  *generative* capstone of that same book.
- **ml-jax-pytorch** (efficiency / systems-ML) — this adds the from-scratch GPU-kernel angle.

## 3. Thesis (the defensible claim)

> **Claim:** "I implemented the full stack of score-based generative modeling — autodiff, a trained
> score network, annealed Langevin sampling, and conditional inpainting — from scratch in C++/CUDA
> with no ML framework, including the CUDA kernels that run the sampler. I can explain and defend
> every line and situate it in the current generative-ML field."

This claim is about **demonstrated capability + understanding + a working artifact.** It does NOT
depend on beating anyone's benchmark. That is deliberate (see §4). Lesson banked from the MoE
project: **do not stake the project on a comparative thesis you cannot cleanly support.**

## 4. Non-goals — what we explicitly do NOT claim

- **NOT** state-of-the-art image quality. Small images, small model, from scratch. Samples should
  be *recognizable*, not photorealistic.
- **NOT** faster / more efficient than PyTorch. The from-scratch kernels are a *learning and
  capability artifact*, not a performance-win claim.
- **NOT** a general ML framework. One model family, one demo, done well.

> Note: an earlier draft framed this around thermodynamic computing / Extropic. Deliberately
> dropped — it made the project read as an Extropic job application rather than a self-justifying
> portfolio piece, and it invited defending a field outside our expertise. The work stands on its
> own ML + systems merits. No thermodynamic/p-bit/quantum framing anywhere in the repo.

## 5. Objectives (concrete, measurable)

Done means all of these exist and are demonstrable:

- **O1.** Reverse-mode **autodiff engine in C++** (tensors, ops, backprop), validated by numerical
  gradient checking to a stated tolerance.
- **O2.** A **hand-written counter-based PRNG** (Philox-style) usable identically on CPU and across
  parallel GPU threads.
- **O3.** A **trained score network** (MLP first, CNN upgrade) trained by our autodiff via
  **denoising score matching** — no framework — that models the image distribution.
- **O4.** Hand-written **CUDA kernels** implementing the sampler inner loop (network forward pass +
  Langevin update), no cuBLAS/cuDNN, validated against the CPU reference.
- **O5.** **Annealed Langevin sampling** producing recognizable unconditional samples.
- **O6.** The **inpainting demo**: masked/corrupted image in, restored image out, via conditional
  sampling (freeze observed pixels, sample the rest). The wow artifact.
- **O7.** **Metrics + writeup**: full metric suite (below), and a writeup that derives the math,
  explains score = −∇ energy and why sampling = MCMC on an energy landscape, and situates the work
  against diffusion / flow matching / consistency models.

## 6. The ML spine (RESOLVED: continuous / score-based — Path B)

> **Score-based generative models ⟺ energy-based models ⟺ MCMC sampling.**

- The model learns a **score** = ∇ₓ log p(x) = **−∇ₓ E(x)** (same object, two names).
- Trained by **denoising score matching** (Vincent 2011): predict the noise that was added; stable,
  no partition-function estimation.
- Sampling = **annealed Langevin dynamics** (Song & Ermon 2019): start from noise, repeatedly step
  along the score plus a little noise, annealing the noise level down.
- **Inpainting = conditional sampling:** run Langevin on the masked region while clamping the
  observed pixels each step, so the fill stays consistent with what's visible.

Concepts owned by end of project (interview surface): score matching & denoising score matching ·
Langevin dynamics & detailed balance · forward/reverse diffusion & noise schedules (DDPM) ·
score-based SDEs · the EBM ↔ score equivalence · and *situated* knowledge of flow matching /
rectified flow and consistency models.

## 7. The path (build ladder — each rung is shippable)

**Rungs 0–5 are the 3–4 week project. Rung 6 finishes it. Rung 4 (CNN) and beyond raise the
ceiling; if time runs short, ship at Rung 3 + a simple inpainting pass and mark the rest future
work.** Model architecture: **MLP-first to get the pipeline green, upgrade to a small CNN for
quality.** Training stays on CPU autodiff; **the CUDA flex is the sampler** (network forward pass +
Langevin update as hand-written kernels).

- **Rung 0 — Scaffold.** CMake + CUDA build, image I/O (PPM / single-header writer), MNIST loader,
  hand-written Philox PRNG (CPU + device). *Artifact:* builds; loads and re-saves a dataset image;
  PRNG passes basic statistical checks.
- **Rung 1 — Autodiff.** Reverse-mode autodiff in C++ (CPU). *Artifact:* gradient check passes.
- **Rung 2 — Trained score net (MLP).** Denoising score matching on grayscale MNIST via our
  autodiff. *Artifact:* loss curve descends; predicted scores match finite-difference on held-out
  samples.
- **Rung 3 — Sampling + CUDA.** Annealed Langevin sampler → unconditional generation. Move the
  sampler inner loop to **hand-written CUDA kernels**; validate GPU vs CPU. *Artifact:* recognizable
  generated digits; GPU sampler matches CPU. **← ML floor; quality risk starts here.**
- **Rung 4 — CNN upgrade.** Replace MLP score net with a small CNN for quality. *Artifact:* visible
  quality jump.
- **Rung 5 — Inpainting.** Conditional sampling with clamped observed pixels. *Artifact:* the
  before/after fill-in-the-hole demo. **← the wow.**
- **Rung 6 — Metrics + writeup.** Full metric suite + the document from O7. *Artifact:* a
  README/writeup a senior eng or researcher would accept.

## 8. Tools, constraints, and boundaries

**Stack:** C++17 host + hand-written CUDA kernels. `nvcc`, CMake. Real NVIDIA GPU + CUDA available.

**Zero ML frameworks** for the learning core (no PyTorch/JAX/TF/cuDNN/cuBLAS). Autodiff, model,
training loop, PRNG, and sampling kernels are all hand-written. This is the flex.

**From-scratch boundary (decided, do not relitigate):**
- **Core = from scratch, no libraries:** autodiff, tensor math, score model, training, PRNG,
  Langevin sampling, CUDA kernels.
- **Allowed thin non-ML utilities:** CMake; a single-header image writer or raw PPM; the C++
  standard library; CUDA runtime (kernel launch / memory only — not cuBLAS/cuDNN).
- **Visualization / benchmarking:** thin Python + matplotlib (or gnuplot) may render charts and
  stitch demo GIFs. Plotting is not the ML and is not part of the from-scratch claim.
- **Evaluation-only exception:** one external **pretrained** model (Inception) is permitted **solely
  to compute FID** for evaluation. It never touches the model, training, or sampling.

**Working conventions (spirit inherited from pml-notes):** Parsa writes the implementation (he is
learning; that is the point). Claude writes tests / gradient-check + kernel-correctness harnesses /
scaffolds / reviews / build config / conventions — not the core impl.

## 9. Dataset

- **Primary: MNIST** (28×28, grayscale). Safest recognizable target.
- **Optional swap: Fashion-MNIST** for a cooler final recording (same size/difficulty).

## 10. Metrics (track everything)

- **Training:** loss curves, gradient norms, per-layer stats.
- **Sampler health:** score-magnitude / energy over steps, autocorrelation, mixing diagnostics.
- **Inpainting fidelity vs ground truth:** PSNR, SSIM, MSE (all hand-coded).
- **Sample realism:** our own small classifier (trained in C++) scoring generated samples; **plus
  FID** via the eval-only pretrained Inception.
- **Eyeball:** sample grids and before/after inpainting panels.

## 11. Showcase pieces (what non-technical people see)

- **Inpainting / restoration** — primary wow ("I hid part of the picture and it filled it back in").
- **Free generation** — grid of samples the model dreamed up.
- **Metrics dashboard** — the credibility layer for technical readers.

## 12. Success criteria (definition of done)

- All of O1–O7 satisfied.
- Every rung's artifact reproducible from a clean clone with documented commands.
- Parsa can, cold, explain any component and the §6 equivalence, and place the work against
  diffusion / flow matching / consistency models.
- Writeup passes: "a senior ML engineer or researcher would accept this as real."
- **Timebox 3–4 weeks.** If short on time, ship at Rung 3 + a basic inpainting pass with an honest
  "future work" section — never ship a broken higher rung.

## 13. Risks and mitigations

- **R1 — Scope vs timebox.** Aggressive while learning. *Mitigation:* rung ladder with a shippable
  floor; CUDA effort concentrated in the sampler; training stays CPU with a small net + small images.
- **R2 — Sample quality underwhelms.** *Mitigation:* denoising score matching is stable (no
  partition function); MLP → CNN upgrade path; lead the demo with inpainting (constrained by
  observed pixels, looks good more easily than free generation).
- **R3 — CUDA training rabbit hole.** *Mitigation:* explicit decision — CUDA flex = sampler, not
  trainer. Train on CPU. Revisit only if ahead of schedule.
- **R4 — Losing the plot (the MoE failure mode).** *Mitigation:* this charter; §3 thesis and §4
  non-goals are re-read before any writeup work.

**Open decisions to confirm before Rung 0:**
1. Project name (`image-inpainting-cpp-cuda` vs `ml-cpp-cuda` vs other).
2. Evaluation: the one eval-only pretrained model for FID is acceptable. (Confirmed.)

## 14. References (papers this project stands on)

- Hyvärinen 2005 — Score Matching.
- Vincent 2011 — Denoising Score Matching.
- Welling & Teh 2011 — Stochastic Gradient Langevin Dynamics.
- Song & Ermon 2019; Song et al. 2021 — Score-based generative modeling / annealed Langevin / SDEs.
- Ho, Jain, Abbeel 2020 — DDPM (situating).
- Lipman et al. 2022; Liu et al. 2022 — Flow Matching / Rectified Flow (situating).
- Song et al. 2023 — Consistency Models (situating).
- Murphy — *Probabilistic Machine Learning: Advanced Topics* (the pml-notes book): EBMs, MCMC,
  diffusion chapters.

## 15. What Parsa will be able to say afterward (the payoff)

- "I wrote reverse-mode autodiff, a counter-based PRNG, and the CUDA sampling kernels by hand; no
  framework touched the core."
- "I can derive why the score is the negative energy gradient, and why sampling is MCMC on an
  energy landscape."
- "My inpainting demo is conditional annealed-Langevin sampling — here's how it relates to DDPM and
  why frontier labs moved to flow matching."
