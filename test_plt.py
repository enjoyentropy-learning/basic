"""
Baseline evaluation of PLTBlock & SymbolicLogicEngine on duplicate detection.

Goal: Can the current architecture (d_model=9, depth=3) learn exact 0/1 outputs
      (zero MSE loss) for detecting whether a 9-symbol sequence has duplicates?

Pipeline:
  1. AdamW training (2500 epochs, float64)
  2. L-BFGS refinement on top
  3. Evaluate on validation set with detailed diagnostics
"""

import torch
import torch.optim as optim

from debugSudoku2 import (
    generate_data,
    SymbolicLogicEngine,
    train_model,
    evaluate_model,
)

# ─── Config ───────────────────────────────────────────────────────────────────
NUM_SYMBOLS = 9
D_MODEL     = 9       # match one-hot dimension
DEPTH       = 3
DTYPE       = torch.float64
TRAIN_FRAC  = 0.014   # ~10K samples (total dataset ~725K)
SEED        = 42

ADAMW_EPOCHS = 2500
ADAMW_LR     = 0.001
ADAMW_LOG    = 500

LBFGS_EPOCHS = 300
LBFGS_LR     = 1.0
LBFGS_LOG    = 50


# ─── Helpers ──────────────────────────────────────────────────────────────────
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def print_gradient_norms(model):
    print("\nGradient norms per parameter group:")
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"  {name:30s}  grad_norm={p.grad.norm().item():.6e}  param_norm={p.norm().item():.6e}")
        else:
            print(f"  {name:30s}  grad=None")


def print_sample_predictions(model, X_oh, y, X_orig, n=10, label=""):
    model.eval()
    with torch.no_grad():
        preds = model(X_oh)
    print(f"\n{label} — Sample predictions (first {n}):")
    print(f"  {'Sequence':<30s} {'Pred':>10s} {'Target':>7s} {'|Err|':>10s}")
    for i in range(min(n, len(y))):
        seq = X_orig[i].tolist()
        p   = preds[i].item()
        t   = y[i].item()
        print(f"  {str(seq):<30s} {p:>10.6f} {t:>7.1f} {abs(p - t):>10.6e}")


def print_prediction_stats(model, X_oh, y, label=""):
    model.eval()
    with torch.no_grad():
        preds = model(X_oh).squeeze()
        targets = y.squeeze()
        errors = (preds - targets).abs()

    print(f"\n{label} — Prediction statistics:")
    print(f"  pred  min={preds.min().item():.6f}  max={preds.max().item():.6f}  "
          f"mean={preds.mean().item():.6f}  std={preds.std().item():.6f}")
    print(f"  error min={errors.min().item():.6e}  max={errors.max().item():.6e}  "
          f"mean={errors.mean().item():.6e}")

    # Accuracy at threshold 0.5
    preds_bin = (preds > 0.5).float()
    acc = (preds_bin == targets).float().mean().item()
    print(f"  accuracy (threshold=0.5): {acc:.6f}")

    # Count predictions in different ranges
    near_0 = (preds < 0.1).sum().item()
    near_1 = (preds > 0.9).sum().item()
    mid    = len(preds) - near_0 - near_1
    print(f"  preds < 0.1: {near_0}  |  0.1 <= preds <= 0.9: {mid}  |  preds > 0.9: {near_1}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("BASELINE: SymbolicLogicEngine on Duplicate Detection")
    print(f"  d_model={D_MODEL}, depth={DEPTH}, dtype={DTYPE}")
    print(f"  train_frac={TRAIN_FRAC}, seed={SEED}")
    print("=" * 70)

    # --- Data ---
    X_train_oh, y_train, X_val_oh, y_val, X_train_orig, X_val_orig = \
        generate_data(num_symbols=NUM_SYMBOLS, train_frac=TRAIN_FRAC, seed=SEED, dtype=DTYPE)

    print(f"\nDataset sizes: train={len(y_train)}, val={len(y_val)}")
    print(f"Train label distribution: valid={y_train.sum().item():.0f}, "
          f"invalid={len(y_train) - y_train.sum().item():.0f}")

    # --- Model ---
    model = SymbolicLogicEngine(vocab_size=D_MODEL, n_layers=DEPTH, dtype=DTYPE)
    loss_fn = torch.nn.MSELoss()

    n_params = count_parameters(model)
    print(f"\nModel parameter count: {n_params}")
    print(f"Model architecture:\n{model}")

    # ── Phase 1: AdamW ────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PHASE 1: AdamW training")
    print("─" * 70)

    optimizer_adamw = optim.AdamW(model.parameters(), lr=ADAMW_LR)
    adamw_history = train_model(
        model, optimizer_adamw, loss_fn, X_train_oh, y_train,
        epochs=ADAMW_EPOCHS, optimizer_type='adamw',
        log_interval=ADAMW_LOG, log_verbose=True,
    )

    # Gradient snapshot after AdamW
    model.train()
    model.zero_grad()
    preds = model(X_train_oh)
    loss = loss_fn(preds, y_train)
    loss.backward()
    print_gradient_norms(model)

    print(f"\nAdamW final train loss: {adamw_history[-1]:.8e}")
    print_prediction_stats(model, X_train_oh, y_train, label="Train (post-AdamW)")
    print_sample_predictions(model, X_train_oh, y_train, X_train_orig, n=10, label="Train (post-AdamW)")

    # ── Phase 1b: Validation after AdamW ─────────────────────────────────
    print("\n" + "─" * 70)
    print("PHASE 1b: Validation evaluation (post-AdamW, pre-LBFGS)")
    print("─" * 70)

    evaluate_model(model, loss_fn, X_val_oh, y_val, X_val_orig, data_name="Validation (post-AdamW)")
    print_prediction_stats(model, X_val_oh, y_val, label="Validation (post-AdamW)")

    # ── Phase 2: L-BFGS refinement ───────────────────────────────────────
    print("\n" + "─" * 70)
    print("PHASE 2: L-BFGS refinement")
    print("─" * 70)

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=LBFGS_LR,
        max_iter=20, history_size=100, line_search_fn="strong_wolfe",
    )
    lbfgs_history = train_model(
        model, optimizer_lbfgs, loss_fn, X_train_oh, y_train,
        epochs=LBFGS_EPOCHS, optimizer_type='lbfgs',
        log_interval=LBFGS_LOG, log_verbose=True,
    )

    # Gradient snapshot after L-BFGS
    model.train()
    model.zero_grad()
    preds = model(X_train_oh)
    loss = loss_fn(preds, y_train)
    loss.backward()
    print_gradient_norms(model)

    print(f"\nL-BFGS final train loss: {lbfgs_history[-1]:.8e}")
    print_prediction_stats(model, X_train_oh, y_train, label="Train (post-LBFGS)")
    print_sample_predictions(model, X_train_oh, y_train, X_train_orig, n=10, label="Train (post-LBFGS)")

    # ── Phase 3: Validation evaluation ───────────────────────────────────
    print("\n" + "─" * 70)
    print("PHASE 3: Validation evaluation")
    print("─" * 70)

    evaluate_model(model, loss_fn, X_val_oh, y_val, X_val_orig, data_name="Validation")
    print_prediction_stats(model, X_val_oh, y_val, label="Validation")
    print_sample_predictions(model, X_val_oh, y_val, X_val_orig, n=10, label="Validation")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Parameters:          {n_params}")
    print(f"  Train samples:       {len(y_train)}")
    print(f"  Val samples:         {len(y_val)}")
    print(f"  AdamW final loss:    {adamw_history[-1]:.8e}")
    print(f"  L-BFGS final loss:   {lbfgs_history[-1]:.8e}")


if __name__ == "__main__":
    main()
