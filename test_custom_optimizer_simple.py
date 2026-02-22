"""
Test the custom optimizer (p -= (loss / ||grad||²) * grad) on simple functions.

This update rule is Newton's root-finding method for f(x)=0, NOT Newton's
optimization method. For scalar f(x) = (x-3)², the update simplifies to
x_new = (x+3)/2, which converges monotonically. Multi-parameter coupling
(shared step_scale = loss/||grad||²) is hypothesized to cause the zig-zag
behavior seen in debugLogicGates.ipynb.

Experiments:
  1. f(x) = (x-3)², scalar
  2. f(x,y) = (x-1)² + (y-2)², symmetric two-param
  3. f(x,y) = (x-1)² + 100*(y-2)², asymmetric two-param (KEY)
  4. Tiny linear network y = w*x + b, two data points
  5. Tiny nonlinear 2-layer MLP

Each experiment runs float32 then float64.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Custom optimizer update (standalone copy from debugLogicGates.py:193-217)
# ============================================================

def custom_gradient_update_params(params, loss, grads):
    """Applies custom gradient update: p -= (loss / ||grad||^2) * grad.

    Works on raw tensors (no model needed). Returns step_scale and grad_norm_sq.
    """
    eps = torch.finfo(loss.dtype).eps
    threshold = eps ** 2

    grad_norm_sq = sum(torch.sum(g ** 2) for g in grads)

    if grad_norm_sq < threshold:
        print(f"  Gradient vanished | grad_norm_sq={grad_norm_sq.item():.3e}")
        return None, grad_norm_sq.item()

    step_scale = loss / grad_norm_sq

    with torch.no_grad():
        for p, g in zip(params, grads):
            p -= step_scale * g

    return step_scale.item(), grad_norm_sq.item()


def custom_gradient_update_model(model, loss):
    """Applies custom gradient update to a torch.nn.Module."""
    eps = torch.finfo(loss.dtype).eps
    threshold = eps ** 2

    grad_norm_sq = torch.tensor(0.0, dtype=loss.dtype)
    for p in model.parameters():
        if p.grad is not None:
            grad_norm_sq += torch.sum(p.grad ** 2)

    if grad_norm_sq < threshold:
        print(f"  Gradient vanished | grad_norm_sq={grad_norm_sq.item():.3e}")
        return None, grad_norm_sq.item()

    step_scale = loss / grad_norm_sq

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= step_scale * p.grad

    return step_scale.item(), grad_norm_sq.item()


# ============================================================
# Experiment 1: f(x) = (x-3)², scalar
# ============================================================

def experiment1(dtype, num_steps=50):
    """Scalar quadratic: f(x) = (x-3)². Theoretical: x_n = 3 + (x0-3)/2^n."""
    print(f"\n{'='*60}")
    print(f"Experiment 1: f(x) = (x-3)², dtype={dtype}")
    print(f"{'='*60}")

    x = torch.tensor(10.0, dtype=dtype, requires_grad=True)
    x0 = x.item()

    history = {'step': [], 'x': [], 'loss': [], 'grad': [],
               'step_scale': [], 'grad_norm_sq': [], 'x_theoretical': []}

    for step in range(num_steps):
        loss = (x - 3.0) ** 2

        if x.grad is not None:
            x.grad.zero_()
        loss.backward()

        x_theoretical = 3.0 + (x0 - 3.0) / (2.0 ** step)

        history['step'].append(step)
        history['x'].append(x.item())
        history['loss'].append(loss.item())
        history['grad'].append(x.grad.item())
        history['x_theoretical'].append(x_theoretical)

        ss, gns = custom_gradient_update_params([x], loss.detach(), [x.grad.detach()])
        history['step_scale'].append(ss)
        history['grad_norm_sq'].append(gns)

        if step < 10 or step % 10 == 0:
            print(f"  step {step:3d}: x={x.item():+.8f}, loss={loss.item():.6e}, "
                  f"grad={history['grad'][-1]:+.6e}, step_scale={ss}")

    return history


def plot_experiment1(h32, h64):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Experiment 1: f(x) = (x-3)², scalar", fontsize=14)

    # x vs step
    ax = axes[0, 0]
    ax.plot(h32['step'], h32['x'], 'b.-', label='float32', markersize=3)
    ax.plot(h64['step'], h64['x'], 'r.--', label='float64', markersize=3)
    ax.plot(h64['step'], h64['x_theoretical'], 'g:', label='theoretical', linewidth=2)
    ax.axhline(y=3.0, color='k', linestyle=':', alpha=0.3, label='target=3')
    ax.set_xlabel('Step')
    ax.set_ylabel('x')
    ax.set_title('x vs step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # loss vs step (log)
    ax = axes[0, 1]
    ax.semilogy(h32['step'], h32['loss'], 'b.-', label='float32', markersize=3)
    ax.semilogy(h64['step'], h64['loss'], 'r.--', label='float64', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs step (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # step_scale vs step
    ax = axes[1, 0]
    ss32 = [s for s in h32['step_scale'] if s is not None]
    ss64 = [s for s in h64['step_scale'] if s is not None]
    steps32 = [h32['step'][i] for i, s in enumerate(h32['step_scale']) if s is not None]
    steps64 = [h64['step'][i] for i, s in enumerate(h64['step_scale']) if s is not None]
    ax.plot(steps32, ss32, 'b.-', label='float32', markersize=3)
    ax.plot(steps64, ss64, 'r.--', label='float64', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('step_scale (loss/||grad||²)')
    ax.set_title('Step scale vs step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # |x - x_theoretical| vs step
    ax = axes[1, 1]
    err32 = [abs(h32['x'][i] - h32['x_theoretical'][i]) for i in range(len(h32['step']))]
    err64 = [abs(h64['x'][i] - h64['x_theoretical'][i]) for i in range(len(h64['step']))]
    ax.semilogy(h32['step'], [e + 1e-45 for e in err32], 'b.-', label='float32', markersize=3)
    ax.semilogy(h64['step'], [e + 1e-320 for e in err64], 'r.--', label='float64', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('|x - x_theoretical|')
    ax.set_title('Deviation from theory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('custom_opt_exp1.png', dpi=150)
    plt.close()
    print("  Saved custom_opt_exp1.png")


# ============================================================
# Experiment 2: f(x,y) = (x-1)² + (y-2)², symmetric
# ============================================================

def experiment2(dtype, num_steps=50):
    """Symmetric two-parameter quadratic."""
    print(f"\n{'='*60}")
    print(f"Experiment 2: f(x,y) = (x-1)² + (y-2)², dtype={dtype}")
    print(f"{'='*60}")

    x = torch.tensor(10.0, dtype=dtype, requires_grad=True)
    y = torch.tensor(10.0, dtype=dtype, requires_grad=True)

    history = {'step': [], 'x': [], 'y': [], 'loss': [],
               'grad_x': [], 'grad_y': [], 'step_scale': [], 'grad_norm_sq': []}

    for step in range(num_steps):
        loss = (x - 1.0) ** 2 + (y - 2.0) ** 2

        if x.grad is not None:
            x.grad.zero_()
        if y.grad is not None:
            y.grad.zero_()
        loss.backward()

        history['step'].append(step)
        history['x'].append(x.item())
        history['y'].append(y.item())
        history['loss'].append(loss.item())
        history['grad_x'].append(x.grad.item())
        history['grad_y'].append(y.grad.item())

        ss, gns = custom_gradient_update_params(
            [x, y], loss.detach(), [x.grad.detach(), y.grad.detach()])
        history['step_scale'].append(ss)
        history['grad_norm_sq'].append(gns)

        if step < 10 or step % 10 == 0:
            print(f"  step {step:3d}: x={x.item():+.6f}, y={y.item():+.6f}, "
                  f"loss={loss.item():.6e}, step_scale={ss}")

    return history


def plot_experiment2(h32, h64):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Experiment 2: f(x,y) = (x-1)² + (y-2)², symmetric", fontsize=14)

    # 2D trajectory with contours
    ax = axes[0, 0]
    xg = np.linspace(-1, 12, 100)
    yg = np.linspace(-1, 12, 100)
    X, Y = np.meshgrid(xg, yg)
    Z = (X - 1) ** 2 + (Y - 2) ** 2
    ax.contour(X, Y, Z, levels=20, alpha=0.5)
    ax.plot(h32['x'], h32['y'], 'b.-', label='float32', markersize=4)
    ax.plot(h64['x'], h64['y'], 'r.--', label='float64', markersize=4)
    ax.plot(1, 2, 'k*', markersize=15, label='optimum')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D trajectory')
    ax.legend()

    # loss vs step
    ax = axes[0, 1]
    ax.semilogy(h32['step'], h32['loss'], 'b.-', label='float32', markersize=3)
    ax.semilogy(h64['step'], h64['loss'], 'r.--', label='float64', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs step (log)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # individual params vs step
    ax = axes[1, 0]
    ax.plot(h32['step'], h32['x'], 'b.-', label='x (f32)', markersize=3)
    ax.plot(h32['step'], h32['y'], 'b.--', label='y (f32)', markersize=3)
    ax.plot(h64['step'], h64['x'], 'r.-', label='x (f64)', markersize=3)
    ax.plot(h64['step'], h64['y'], 'r.--', label='y (f64)', markersize=3)
    ax.axhline(y=1.0, color='blue', linestyle=':', alpha=0.3)
    ax.axhline(y=2.0, color='green', linestyle=':', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Parameter value')
    ax.set_title('Parameters vs step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # step_scale
    ax = axes[1, 1]
    ss32 = [s for s in h32['step_scale'] if s is not None]
    ss64 = [s for s in h64['step_scale'] if s is not None]
    steps32 = [h32['step'][i] for i, s in enumerate(h32['step_scale']) if s is not None]
    steps64 = [h64['step'][i] for i, s in enumerate(h64['step_scale']) if s is not None]
    ax.plot(steps32, ss32, 'b.-', label='float32', markersize=3)
    ax.plot(steps64, ss64, 'r.--', label='float64', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('step_scale')
    ax.set_title('Step scale vs step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('custom_opt_exp2.png', dpi=150)
    plt.close()
    print("  Saved custom_opt_exp2.png")


# ============================================================
# Experiment 3: f(x,y) = (x-1)² + 100*(y-2)², asymmetric (KEY)
# ============================================================

def experiment3(dtype, num_steps=100):
    """Asymmetric two-parameter quadratic. The y gradient dominates ||grad||²
    early on, starving x updates. This is the KEY experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment 3: f(x,y) = (x-1)² + 100*(y-2)², dtype={dtype}")
    print(f"{'='*60}")

    x = torch.tensor(10.0, dtype=dtype, requires_grad=True)
    y = torch.tensor(10.0, dtype=dtype, requires_grad=True)

    history = {'step': [], 'x': [], 'y': [], 'loss': [],
               'grad_x': [], 'grad_y': [], 'step_scale': [], 'grad_norm_sq': [],
               'grad_ratio': []}

    for step in range(num_steps):
        loss = (x - 1.0) ** 2 + 100.0 * (y - 2.0) ** 2

        if x.grad is not None:
            x.grad.zero_()
        if y.grad is not None:
            y.grad.zero_()
        loss.backward()

        gx = abs(x.grad.item())
        gy = abs(y.grad.item())
        ratio = gx / gy if gy > 1e-30 else float('inf')

        history['step'].append(step)
        history['x'].append(x.item())
        history['y'].append(y.item())
        history['loss'].append(loss.item())
        history['grad_x'].append(x.grad.item())
        history['grad_y'].append(y.grad.item())
        history['grad_ratio'].append(ratio)

        ss, gns = custom_gradient_update_params(
            [x, y], loss.detach(), [x.grad.detach(), y.grad.detach()])
        history['step_scale'].append(ss)
        history['grad_norm_sq'].append(gns)

        if step < 15 or step % 10 == 0:
            print(f"  step {step:3d}: x={x.item():+.8f}, y={y.item():+.8f}, "
                  f"loss={loss.item():.6e}, |gx/gy|={ratio:.4f}, step_scale={ss}")

    # Check for zig-zag: did loss ever increase?
    increases = []
    for i in range(1, len(history['loss'])):
        if history['loss'][i] > history['loss'][i-1]:
            increases.append((i, history['loss'][i-1], history['loss'][i]))
    if increases:
        print(f"\n  *** ZIG-ZAG DETECTED: loss increased at {len(increases)} steps ***")
        for step_i, prev, curr in increases[:5]:
            print(f"      step {step_i}: {prev:.6e} -> {curr:.6e} (increase: {curr-prev:.6e})")
    else:
        print(f"\n  No zig-zag: loss decreased monotonically across all {num_steps} steps.")

    return history


def plot_experiment3(h32, h64):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Experiment 3: f(x,y) = (x-1)² + 100*(y-2)², ASYMMETRIC (KEY)", fontsize=14)

    # 2D trajectory with contours
    ax = axes[0, 0]
    xg = np.linspace(-2, 12, 200)
    yg = np.linspace(-1, 12, 200)
    X, Y = np.meshgrid(xg, yg)
    Z = (X - 1) ** 2 + 100 * (Y - 2) ** 2
    ax.contour(X, Y, Z, levels=np.logspace(0, 4, 30), alpha=0.5)
    ax.plot(h32['x'], h32['y'], 'b.-', label='float32', markersize=4)
    ax.plot(h64['x'], h64['y'], 'r.--', label='float64', markersize=4)
    ax.plot(1, 2, 'k*', markersize=15, label='optimum')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D trajectory')
    ax.legend()

    # loss vs step
    ax = axes[0, 1]
    ax.semilogy(h32['step'], h32['loss'], 'b.-', label='float32', markersize=3)
    ax.semilogy(h64['step'], h64['loss'], 'r.--', label='float64', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs step (log)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # x vs step
    ax = axes[0, 2]
    ax.plot(h32['step'], h32['x'], 'b.-', label='x (f32)', markersize=3)
    ax.plot(h64['step'], h64['x'], 'r.--', label='x (f64)', markersize=3)
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.3, label='target=1')
    ax.set_xlabel('Step')
    ax.set_ylabel('x')
    ax.set_title('x (small-gradient param) vs step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # y vs step
    ax = axes[1, 0]
    ax.plot(h32['step'], h32['y'], 'b.-', label='y (f32)', markersize=3)
    ax.plot(h64['step'], h64['y'], 'r.--', label='y (f64)', markersize=3)
    ax.axhline(y=2.0, color='k', linestyle=':', alpha=0.3, label='target=2')
    ax.set_xlabel('Step')
    ax.set_ylabel('y')
    ax.set_title('y (large-gradient param) vs step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # |grad_x| / |grad_y| ratio
    ax = axes[1, 1]
    ax.semilogy(h32['step'], h32['grad_ratio'], 'b.-', label='float32', markersize=3)
    ax.semilogy(h64['step'], h64['grad_ratio'], 'r.--', label='float64', markersize=3)
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('|grad_x| / |grad_y|')
    ax.set_title('Gradient ratio (< 1 means y dominates)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # step_scale
    ax = axes[1, 2]
    ss32 = [s for s in h32['step_scale'] if s is not None]
    ss64 = [s for s in h64['step_scale'] if s is not None]
    steps32 = [h32['step'][i] for i, s in enumerate(h32['step_scale']) if s is not None]
    steps64 = [h64['step'][i] for i, s in enumerate(h64['step_scale']) if s is not None]
    ax.plot(steps32, ss32, 'b.-', label='float32', markersize=3)
    ax.plot(steps64, ss64, 'r.--', label='float64', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('step_scale')
    ax.set_title('Step scale vs step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('custom_opt_exp3.png', dpi=150)
    plt.close()
    print("  Saved custom_opt_exp3.png")


# ============================================================
# Experiment 4: Tiny linear network y = w*x + b
# ============================================================

def experiment4(dtype, num_steps=100):
    """Linear network fitting {(1,2), (2,5)} → exact solution w=3, b=-1."""
    print(f"\n{'='*60}")
    print(f"Experiment 4: y = w*x + b, fitting (1,2) & (2,5), dtype={dtype}")
    print(f"{'='*60}")

    w = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    b = torch.tensor(0.0, dtype=dtype, requires_grad=True)

    x_data = torch.tensor([1.0, 2.0], dtype=dtype)
    y_data = torch.tensor([2.0, 5.0], dtype=dtype)

    history = {'step': [], 'w': [], 'b': [], 'loss': [],
               'grad_w': [], 'grad_b': [], 'step_scale': [], 'grad_norm_sq': []}

    for step in range(num_steps):
        y_pred = w * x_data + b
        loss = ((y_pred - y_data) ** 2).mean()

        if w.grad is not None:
            w.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()
        loss.backward()

        history['step'].append(step)
        history['w'].append(w.item())
        history['b'].append(b.item())
        history['loss'].append(loss.item())
        history['grad_w'].append(w.grad.item())
        history['grad_b'].append(b.grad.item())

        ss, gns = custom_gradient_update_params(
            [w, b], loss.detach(), [w.grad.detach(), b.grad.detach()])
        history['step_scale'].append(ss)
        history['grad_norm_sq'].append(gns)

        if step < 15 or step % 10 == 0:
            print(f"  step {step:3d}: w={w.item():+.8f}, b={b.item():+.8f}, "
                  f"loss={loss.item():.6e}, step_scale={ss}")

    # Check for zig-zag
    increases = []
    for i in range(1, len(history['loss'])):
        if history['loss'][i] > history['loss'][i-1]:
            increases.append((i, history['loss'][i-1], history['loss'][i]))
    if increases:
        print(f"\n  *** ZIG-ZAG DETECTED: loss increased at {len(increases)} steps ***")
        for step_i, prev, curr in increases[:5]:
            print(f"      step {step_i}: {prev:.6e} -> {curr:.6e}")
    else:
        print(f"\n  No zig-zag: loss decreased monotonically.")

    return history


def plot_experiment4(h32, h64):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Experiment 4: y = w*x + b, fitting {(1,2),(2,5)}", fontsize=14)

    # 2D trajectory with contours
    ax = axes[0, 0]
    wg = np.linspace(-2, 5, 200)
    bg = np.linspace(-4, 3, 200)
    W, B = np.meshgrid(wg, bg)
    # MSE for (1,2) and (2,5): ((w+b-2)² + (2w+b-5)²) / 2
    Z = ((W + B - 2) ** 2 + (2 * W + B - 5) ** 2) / 2
    ax.contour(W, B, Z, levels=np.logspace(-2, 2, 30), alpha=0.5)
    ax.plot(h32['w'], h32['b'], 'b.-', label='float32', markersize=4)
    ax.plot(h64['w'], h64['b'], 'r.--', label='float64', markersize=4)
    ax.plot(3, -1, 'k*', markersize=15, label='optimum (3,-1)')
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_title('(w, b) trajectory')
    ax.legend()

    # loss vs step
    ax = axes[0, 1]
    ax.semilogy(h32['step'], h32['loss'], 'b.-', label='float32', markersize=3)
    ax.semilogy(h64['step'], h64['loss'], 'r.--', label='float64', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs step (log)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # w, b vs step
    ax = axes[1, 0]
    ax.plot(h32['step'], h32['w'], 'b.-', label='w (f32)', markersize=3)
    ax.plot(h32['step'], h32['b'], 'b.--', label='b (f32)', markersize=3)
    ax.plot(h64['step'], h64['w'], 'r.-', label='w (f64)', markersize=3)
    ax.plot(h64['step'], h64['b'], 'r.--', label='b (f64)', markersize=3)
    ax.axhline(y=3.0, color='blue', linestyle=':', alpha=0.3)
    ax.axhline(y=-1.0, color='green', linestyle=':', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Parameter value')
    ax.set_title('w, b vs step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # step_scale
    ax = axes[1, 1]
    ss32 = [s for s in h32['step_scale'] if s is not None]
    ss64 = [s for s in h64['step_scale'] if s is not None]
    steps32 = [h32['step'][i] for i, s in enumerate(h32['step_scale']) if s is not None]
    steps64 = [h64['step'][i] for i, s in enumerate(h64['step_scale']) if s is not None]
    ax.plot(steps32, ss32, 'b.-', label='float32', markersize=3)
    ax.plot(steps64, ss64, 'r.--', label='float64', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('step_scale')
    ax.set_title('Step scale vs step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('custom_opt_exp4.png', dpi=150)
    plt.close()
    print("  Saved custom_opt_exp4.png")


# ============================================================
# Experiment 5: Tiny nonlinear 2-layer MLP
# ============================================================

class TinyMLP(torch.nn.Module):
    """2-layer MLP: input(1) -> hidden(4) -> output(1), with x² activation."""
    def __init__(self, dtype):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 4, dtype=dtype)
        self.fc2 = torch.nn.Linear(4, 1, dtype=dtype)

    def forward(self, x):
        h = self.fc1(x)
        h = h ** 2  # polynomial activation (like PLT)
        return self.fc2(h)


def experiment5(dtype, num_steps=200):
    """Nonlinear MLP fitting a few data points. Tests non-convexity effects."""
    print(f"\n{'='*60}")
    print(f"Experiment 5: Tiny MLP with x² activation, dtype={dtype}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    model = TinyMLP(dtype)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params}")

    # Fit 4 data points: y = x²
    x_data = torch.tensor([[-1.0], [0.0], [1.0], [2.0]], dtype=dtype)
    y_data = torch.tensor([[1.0], [0.0], [1.0], [4.0]], dtype=dtype)

    history = {'step': [], 'loss': [], 'step_scale': [], 'grad_norm_sq': [],
               'param_norms': []}

    for step in range(num_steps):
        y_pred = model(x_data)
        loss = ((y_pred - y_data) ** 2).mean()

        model.zero_grad()
        loss.backward()

        pnorm = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5

        history['step'].append(step)
        history['loss'].append(loss.item())
        history['param_norms'].append(pnorm)

        ss, gns = custom_gradient_update_model(model, loss.detach())
        history['step_scale'].append(ss)
        history['grad_norm_sq'].append(gns)

        if step < 15 or step % 20 == 0:
            print(f"  step {step:3d}: loss={loss.item():.6e}, "
                  f"||params||={pnorm:.4f}, step_scale={ss}")

    # Check for zig-zag
    increases = []
    for i in range(1, len(history['loss'])):
        if history['loss'][i] > history['loss'][i-1]:
            increases.append((i, history['loss'][i-1], history['loss'][i]))
    if increases:
        print(f"\n  *** ZIG-ZAG DETECTED: loss increased at {len(increases)} steps ***")
        for step_i, prev, curr in increases[:10]:
            print(f"      step {step_i}: {prev:.6e} -> {curr:.6e}")
    else:
        print(f"\n  No zig-zag: loss decreased monotonically.")

    return history


def plot_experiment5(h32, h64):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Experiment 5: Tiny MLP with x² activation", fontsize=14)

    # loss vs step
    ax = axes[0, 0]
    ax.semilogy(h32['step'], h32['loss'], 'b.-', label='float32', markersize=3)
    ax.semilogy(h64['step'], h64['loss'], 'r.--', label='float64', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs step (log)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # param norm vs step
    ax = axes[0, 1]
    ax.plot(h32['step'], h32['param_norms'], 'b.-', label='float32', markersize=3)
    ax.plot(h64['step'], h64['param_norms'], 'r.--', label='float64', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('||params||')
    ax.set_title('Parameter norm vs step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # step_scale
    ax = axes[1, 0]
    ss32 = [s for s in h32['step_scale'] if s is not None]
    ss64 = [s for s in h64['step_scale'] if s is not None]
    steps32 = [h32['step'][i] for i, s in enumerate(h32['step_scale']) if s is not None]
    steps64 = [h64['step'][i] for i, s in enumerate(h64['step_scale']) if s is not None]
    if ss32:
        ax.semilogy(steps32, ss32, 'b.-', label='float32', markersize=3)
    if ss64:
        ax.semilogy(steps64, ss64, 'r.--', label='float64', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('step_scale')
    ax.set_title('Step scale vs step (log)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # grad_norm_sq
    ax = axes[1, 1]
    ax.semilogy(h32['step'], h32['grad_norm_sq'], 'b.-', label='float32', markersize=3)
    ax.semilogy(h64['step'], h64['grad_norm_sq'], 'r.--', label='float64', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('||grad||²')
    ax.set_title('Gradient norm squared vs step (log)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('custom_opt_exp5.png', dpi=150)
    plt.close()
    print("  Saved custom_opt_exp5.png")


# ============================================================
# Main
# ============================================================

def main():
    print("Custom Optimizer Test: p -= (loss / ||grad||²) * grad")
    print("=" * 60)

    # Experiment 1
    h1_32 = experiment1(torch.float32)
    h1_64 = experiment1(torch.float64)
    plot_experiment1(h1_32, h1_64)

    # Experiment 2
    h2_32 = experiment2(torch.float32)
    h2_64 = experiment2(torch.float64)
    plot_experiment2(h2_32, h2_64)

    # Experiment 3 (KEY)
    h3_32 = experiment3(torch.float32)
    h3_64 = experiment3(torch.float64)
    plot_experiment3(h3_32, h3_64)

    # Experiment 4
    h4_32 = experiment4(torch.float32)
    h4_64 = experiment4(torch.float64)
    plot_experiment4(h4_32, h4_64)

    # Experiment 5
    h5_32 = experiment5(torch.float32)
    h5_64 = experiment5(torch.float64)
    plot_experiment5(h5_32, h5_64)

    print("\n" + "=" * 60)
    print("All experiments complete. Plots saved:")
    print("  custom_opt_exp1.png — Scalar quadratic")
    print("  custom_opt_exp2.png — Symmetric 2-param")
    print("  custom_opt_exp3.png — Asymmetric 2-param (KEY)")
    print("  custom_opt_exp4.png — Linear network")
    print("  custom_opt_exp5.png — Nonlinear MLP")


if __name__ == "__main__":
    main()
