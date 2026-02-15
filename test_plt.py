"""
Experiment 5: More data + more epochs + cosine LR schedule.

Changes from Exp 3:
  - train_frac=0.14 (~100K samples, up from 10K)
  - 5000 epochs (up from 2500)
  - Cosine annealing LR schedule

Architecture (unchanged from Exp 3):
  - Flatten pooling, attention scaling by 1/sqrt(d_model)
  - d_model=9, depth=3, no embedding
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from debugSudoku2 import (
    generate_data,
    SymbolicLogicEngine,
    evaluate_model,
)

# ─── Config ───────────────────────────────────────────────────────────────────
NUM_SYMBOLS = 9
D_MODEL     = 9       # match one-hot dimension (no embedding)
DEPTH       = 3
DTYPE       = torch.float64
TRAIN_FRAC  = 0.14    # ~100K samples (total dataset ~725K)
SEED        = 42

ADAMW_EPOCHS = 5000
ADAMW_LR     = 0.001
ADAMW_LOG    = 500


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

    preds_bin = (preds > 0.5).float()
    acc = (preds_bin == targets).float().mean().item()
    print(f"  accuracy (threshold=0.5): {acc:.6f}")

    near_0 = (preds < 0.1).sum().item()
    near_1 = (preds > 0.9).sum().item()
    mid    = len(preds) - near_0 - near_1
    print(f"  preds < 0.1: {near_0}  |  0.1 <= preds <= 0.9: {mid}  |  preds > 0.9: {near_1}")


# ─── Training loop with cosine annealing ─────────────────────────────────────
def train_with_cosine(model, loss_fn, X_train_oh, y_train, epochs, lr, log_interval):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    loss_history = []
    N = X_train_oh.size(0)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_train_oh)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        loss_history.append(current_loss)

        if epoch % log_interval == 0:
            with torch.no_grad():
                preds_bin = (preds > 0.5).float()
                acc = (preds_bin == y_train).float().mean().item()
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:5d} | "
                f"Loss: {current_loss:.8f} | "
                f"Acc: {acc:.8f} | "
                f"LR: {lr_now:.6e}"
            )

    return loss_history


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Exp 5: More Data + More Epochs + Cosine LR")
    print(f"  d_model={D_MODEL}, depth={DEPTH}, dtype={DTYPE}")
    print(f"  train_frac={TRAIN_FRAC}, seed={SEED}")
    print(f"  epochs={ADAMW_EPOCHS}, lr={ADAMW_LR}, cosine→1e-6")
    print("=" * 70)

    # --- Data ---
    X_train_oh, y_train, X_val_oh, y_val, X_train_orig, X_val_orig = \
        generate_data(num_symbols=NUM_SYMBOLS, train_frac=TRAIN_FRAC, seed=SEED, dtype=DTYPE)

    print(f"\nDataset sizes: train={len(y_train)}, val={len(y_val)}")
    print(f"Train label distribution: valid={y_train.sum().item():.0f}, "
          f"invalid={len(y_train) - y_train.sum().item():.0f}")

    # --- Model ---
    model = SymbolicLogicEngine(vocab_size=NUM_SYMBOLS, seq_len=NUM_SYMBOLS, d_model=D_MODEL, n_layers=DEPTH, dtype=DTYPE)
    loss_fn = torch.nn.MSELoss()

    n_params = count_parameters(model)
    print(f"\nModel parameter count: {n_params}")
    print(f"Model architecture:\n{model}")

    # ── Training ──────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("AdamW + Cosine Annealing")
    print("─" * 70)

    history = train_with_cosine(
        model, loss_fn, X_train_oh, y_train,
        epochs=ADAMW_EPOCHS, lr=ADAMW_LR, log_interval=ADAMW_LOG,
    )

    # Gradient snapshot
    model.train()
    model.zero_grad()
    preds = model(X_train_oh)
    loss = loss_fn(preds, y_train)
    loss.backward()
    print_gradient_norms(model)

    print(f"\nFinal train loss: {history[-1]:.8e}")
    print_prediction_stats(model, X_train_oh, y_train, label="Train")
    print_sample_predictions(model, X_train_oh, y_train, X_train_orig, n=10, label="Train")

    # ── Validation ────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Validation evaluation")
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
    print(f"  Final train loss:    {history[-1]:.8e}")


if __name__ == "__main__":
    main()
