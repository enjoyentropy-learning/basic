# Project: Symbolic Logic Engine for Duplicate Detection

## Goal
Evaluate whether PLTBlock and SymbolicLogicEngine can achieve exact 0/1 outputs (zero MSE loss) on duplicate detection with 9 symbols, one-hot encoded. Valid permutations → 1, sequences with duplicates → 0.

## Setup
- Python 3.13.3, PyTorch 2.10.0
- Virtual environment at `.venv/` — always activate before running: `source .venv/bin/activate`
- Run test script: `source .venv/bin/activate && python test_plt.py 2>&1 | tee test_plt_output_v<N>.log`

## Key Files
- `debugSudoku2.py` — Main source: data generation, PLTBlock, SymbolicLogicEngine, training loops, evaluation
- `test_plt.py` — Current test script (Experiment 5 config: 100K data, 5000 epochs, cosine LR)
- `experiment_log.md` — Full experiment history with results and observations
- `test_plt_output_v*.log` — Raw output logs from each experiment run

## Architecture: SymbolicLogicEngine
- PLTBlock: raw polynomial attention (Q*K^T/sqrt(d_model) * V + proj + residual), no activations
- d_model=9 (matches one-hot vocab size), depth=3 layers
- Flatten pooling (preserves positional structure) → Linear(81→1) readout
- 1,081 parameters total
- Outputs are unbounded (no sigmoid/clamp) — this is intentional for "symbolic" purity

## Changes Made to debugSudoku2.py (from original)
1. **PLTBlock**: Added attention scaling by `1/sqrt(d_model)` to prevent magnitude explosion
2. **SymbolicLogicEngine**: Replaced `x.mean(dim=1)` pooling with `x.flatten(start_dim=1)` to preserve all position-symbol information. Readout changed from Linear(9→1) to Linear(81→1).
3. **SymbolicLogicEngine**: Added optional `d_model` and `seq_len` params, plus optional embedding layer `Linear(vocab_size→d_model)` (currently unused — d_model=9 without embedding works best)

## Current Best Result (Experiment 5)
- Config: d_model=9, depth=3, float64, train_frac=0.14 (~100K samples), AdamW lr=0.001, 5000 epochs, cosine annealing LR → 1e-6, full-batch
- Train loss: 8.73e-04 (never reaches zero)
- Val loss: 0.037, Val acc: 99.996% (only 26 errors out of 624K)
- Val pred range: [-91.4, +19.7]
- Training time: ~2.8 hours on CPU (each epoch ~2s with 100K data)

## Key Theoretical Insight
det(X)^2 = 1 for valid permutation matrices, 0 for matrices with duplicate rows. This is a degree-18 polynomial in the input entries. The PLT architecture with 3 layers can represent up to degree-27 polynomials, so the architecture CAN theoretically express the exact solution. The challenge is optimization finding it.

## Remaining Problems
1. **Train loss plateaus at ~8.7e-04** — gradients become tiny (~1e-4), stuck at local minimum, never reaches zero
2. **Polynomial blowup on extreme inputs** — 26 val errors all on sequences with heavy symbol repetition (e.g. [3,3,2,3,2,2,3,1,2]), predictions go to -91 or +20
3. **Predictions not exactly 0/1** — valid sequences predict ~0.999, invalid ~-0.03 to 0.02

## Things That Helped
- Flatten pooling (over mean pooling) — preserves positional info, halved val errors
- Attention scaling — prevents magnitude explosion through layers
- More training data — biggest single lever (val loss: 9.6 at 10K → 0.037 at 100K)
- More epochs + cosine LR — loss still decreasing, cosine helps convergence

## Things That Didn't Help
- L-BFGS refinement — overfits training data, destroys validation (100x worse val loss)
- Embedding layer (d_model=32) — 12x more params but 55x worse val loss, overfits
- See `experiment_log.md` for full details

## Potential Next Steps to Explore
- Stronger weight decay in AdamW (currently default 0.01) — penalize large weights causing polynomial blowup
- Even more training data (train on full dataset ~725K) — 1081 params << 725K samples, would prove if architecture can represent the exact function
- Different initialization (Xavier/orthogonal)
- Gradient clipping
- Architecture changes: remove/modify residual connections, try different aggregation
