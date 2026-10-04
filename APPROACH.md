# GFAP Spatial Moment Reconstruction — approach

## The reframing

Every one of the 64 targets is defined as a sum of per-cell contributions over the
cells whose transformed box centre falls inside a terminal territory. Each number
is therefore **an integral of a per-cell burden field over a rectangle**.

That has three consequences the solution is built around:

1. **The network never sees the territory program.** A from-scratch convolutional
   encoder–decoder maps one microscopy field to four non-negative burden-density
   maps at stride 4 (72×96). The published rectangle geometry is applied
   afterwards. Nothing about the 16 territories enters the model.
2. **Integration is exactly separable.** A leaf value is `wyᵀ · M · wx`, where
   `wy`/`wx` are 1-D pixel-overlap fractions. It is exact, differentiable and
   costs almost nothing, so all four programs of a field are scored from a
   single encode.
3. **The hierarchy is consistent by construction.** Because integration is
   additive, `parent = left + right` holds at every node automatically — there is
   no way for the 31-node reconstruction to disagree with the 16 terminals.

Measured ceiling of this representation at stride 4: **90.4**.

## Training signal: fusing the four programs

Sixteen coarse sums is weak supervision for a dense map. But each field is
partitioned four *different* ways. Overlaying the four partitions yields a common
refinement of ~637 cells (~15×12 px) — about 40× finer than any single territory.
Iterative proportional fitting on that refinement recovers the maximum-entropy
burden field consistent with **all four programs simultaneously**, and supervises
the density maps directly.

This is a *scaffold*, not the objective. Max-entropy fitting spreads each cell's
mass along whichever axis the constraints do not pin, producing streak artifacts
a CNN cannot reproduce from image evidence. Measured on a held-out field split:

| dense-target schedule                | score |
|--------------------------------------|-------|
| constant weight                      | 23.6  |
| annealed to zero by epoch 18          | 25.7  |
| annealed away, then pure rectangle loss | **27.0** |

So the scaffold is annealed off and the exact rectangle-integral objective —
which mirrors the published scoring formula (root term, depth-weighted tree term,
terminal-distribution term, at the published weights) — drives the rest of
training.

## The diagnostic that redirected the work

Substituting the **true per-field totals** into the model's own predicted spatial
shape moved the score from **27.0 → 36.0**, while perfect shape with the model's
own totals gave 58.1. Separately, blurring a perfect density by σ = 32 px still
scores 43.8.

Localisation precision was never the bottleneck — **counting was**. The cause is
structural: the per-territory objective divides each error by `(true + floor)`,
which punishes over-stating an empty territory far more than under-stating an
occupied one, and so pulls an integrated density systematically low (predicted
field total 9.9 against a true 13.4).

The fix is to **decouple "how many" from "where"**: a globally pooled head
predicts the four field totals, and the spatial map contributes only a normalised
distribution. On a matched held-out split this lifted the score at every
checkpoint (epoch 5: 18.8 vs 16.2; epoch 11: 22.8 vs 21.0; converged: 28.1).

## Output calibration

Because the score is not maximised by the conditional mean, a three-parameter
monotone map per channel — dead-zone, exponent, gain — is fitted out-of-fold
against the published formula. A 10-knot free monotone map was tested and gave
the *same* in-sample result (27.45 vs 27.46), confirming three parameters
saturate the available gain.

## Other measured facts

- All four channels are near-proportional to count (per-cell ratios
  1.233 / 1.145 / 0.730), so channels 1–3 are parameterised as count × a locally
  predicted ratio.
- Train and test program geometry distributions are identical — no shift.
- Oracle root with a uniform spread scores only 19.1, so totals alone are far
  from sufficient.
- Re-running the fitting with the network's own density as prior sharpens the
  targets (ceiling 90.4 → 92.4) and beats plain max-entropy (25.6 vs 24.8), but
  still loses to using no dense loss at all after warm-up — so it is not used.

## Compliance

All trainable parameters start from random initialisation. No pretrained weights,
no external data, no foundation-model features. The learned density field is the
sole predictor; the program fusion is a training-target construction derived only
from the released training labels, as permitted. Validation is grouped by
`field_id` throughout — all four rows of a field always stay on the same side of
a split. Dependencies are numpy, torch and PIL only.

`solution.py` runs as `python3 solution.py <public_dir> <submission_out.csv>`,
times a few real training steps up front, and sizes the number of epochs and
independently initialised models to fit its wall-clock budget on a CPU-only box.
