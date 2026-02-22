# Custom Optimizer Research: `p -= (loss / ||grad||²) * grad`

## Update Rule

Given parameters `p`, loss `L`, and gradient `g = ∂L/∂p`:

```
step_scale = L / ||g||²
p_new = p - step_scale * g
```

Key property: **all parameters share the same scalar `step_scale`**. This is fundamentally different from per-parameter adaptive methods like Adam.

## Theoretical Analysis

### Newton's Root-Finding Interpretation

This update is **Newton's method for finding roots of f(x) = 0**, not Newton's optimization method for finding minima. The distinction matters:

- Newton's optimization: `x -= f'(x) / f''(x)` (uses Hessian, finds where gradient = 0)
- Newton's root-finding: `x -= f(x) / f'(x)` (finds where function = 0)
- Our rule: `p -= f(p) / ||f'(p)||²` (multivariate generalization, projects onto gradient direction)

### Scalar Case: f(x) = (x - a)²

For a single parameter with quadratic loss:

```
g = 2(x - a)
||g||² = 4(x - a)²
step_scale = (x - a)² / (4(x - a)²) = 1/4
x_new = x - (1/4) * 2(x - a) = x - (x - a)/2 = (x + a) / 2
```

This gives **x_n = a + (x₀ - a) / 2ⁿ** — monotonic convergence with rate 1/2. No zig-zag possible.

### Symmetric Multi-Parameter Case: f(x,y) = (x-a)² + (y-b)²

```
gx = 2(x-a), gy = 2(y-b)
||g||² = 4(x-a)² + 4(y-b)²
step_scale = [(x-a)² + (y-b)²] / [4(x-a)² + 4(y-b)²] = 1/4
```

Same as scalar: each parameter gets `p -= (1/4) * 2(p - target) = p - (p - target)/2`. Monotonic.

### Asymmetric Case: f(x,y) = (x-a)² + c*(y-b)² with c >> 1 (KEY)

```
gx = 2(x-a), gy = 2c(y-b)
||g||² = 4(x-a)² + 4c²(y-b)²
step_scale = [(x-a)² + c(y-b)²] / [4(x-a)² + 4c²(y-b)²]
```

When `c(y-b)² >> (x-a)²` (early training, y-gradient dominates):
```
step_scale ≈ c(y-b)² / (4c²(y-b)²) = 1/(4c)
```
- y update: `y -= (1/(4c)) * 2c(y-b) = y - (y-b)/2` → rate 1/2 (good)
- x update: `x -= (1/(4c)) * 2(x-a) = x - (x-a)/(2c)` → rate 1/(2c) (STARVED)

With c=100, the x-parameter converges 100x slower than y. Once y converges:
```
step_scale ≈ (x-a)² / (4(x-a)²) = 1/4
```
x converges at rate 1/2 again, but only after y has finished.

**The shared step_scale creates a sequential convergence pattern**: dominant-gradient parameters converge first, starving the rest.

### Can This Cause Zig-Zag?

For pure quadratics: **no zig-zag**, only slow convergence for starved parameters. The loss decreases monotonically because each step is a Newton root-finding step projected onto the gradient.

However, for **non-quadratic losses** (neural networks):
1. The loss landscape curvature changes as parameters update
2. The shared step_scale may overshoot some parameters while undershoot others
3. After dominant parameters settle, previously-starved parameters suddenly get full step_scale and may overshoot
4. This creates **oscillation between parameter subspaces** → zig-zag in total loss

## Multi-Parameter Coupling Hypothesis

The zig-zag in `debugLogicGates.ipynb` arises because:

1. PLT attention creates **highly non-uniform gradient magnitudes** across layers
2. Early layers (near input) have large gradients → dominate `||grad||²`
3. Later layers get starved updates → accumulate error
4. When early layers converge, step_scale suddenly increases → later layers overshoot
5. Overshoot in later layers increases loss → loss goes up → oscillation

This is fundamentally a **gradient norm coupling problem**: the single scalar `step_scale` cannot accommodate parameters with different curvatures.

## Experiments

### Experiment 1: f(x) = (x-3)², scalar

**Purpose**: Verify the theoretical formula x_n = 3 + (x₀-3)/2ⁿ matches implementation.

**Expected**: Monotonic convergence, no zig-zag. Float32 and float64 should agree until float32 precision limit (~7 decimal digits).

**Results**: *(to be filled after running)*

---

### Experiment 2: f(x,y) = (x-1)² + (y-2)², symmetric

**Purpose**: Confirm symmetric multi-parameter case also converges monotonically.

**Expected**: Both params converge at rate 1/2, identical to scalar case. Straight-line trajectory in (x,y) space from (10,10) to (1,2).

**Results**: *(to be filled after running)*

---

### Experiment 3: f(x,y) = (x-1)² + 100*(y-2)², asymmetric — KEY

**Purpose**: Demonstrate the parameter starvation effect with asymmetric curvature.

**Expected**:
- y converges quickly (rate ~1/2), x converges slowly (rate ~1/200)
- No zig-zag (still quadratic), but dramatically different convergence rates
- |grad_x|/|grad_y| ratio starts near 9/1600 ≈ 0.006, grows toward 1 as y settles
- Loss decreases monotonically but with a "kink" when y finishes and x takes over

**Results**: *(to be filled after running)*

---

### Experiment 4: Linear network y = w*x + b

**Purpose**: Test with a loss function that couples parameters through data (MSE creates cross-terms in the gradient).

**Expected**: Possible mild zig-zag due to w-b coupling through data matrix. The loss landscape is still quadratic (linear model + MSE), so theoretically should converge monotonically, but coupling may create issues.

**Results**: *(to be filled after running)*

---

### Experiment 5: Tiny nonlinear MLP with x² activation

**Purpose**: Test the full hypothesis — non-convexity + shared step_scale → zig-zag.

**Expected**: Most likely to show zig-zag. The x² activation creates a non-convex loss landscape where the shared step_scale cannot appropriately scale all parameters simultaneously.

**Results**: *(to be filled after running)*

---

## Connection to PLT / Duplicate Detection

The SymbolicLogicEngine's PLT attention creates polynomial mappings of degree up to 3^depth. With depth=3, outputs are degree-27 polynomials of the inputs. This means:

1. **Gradient magnitudes vary enormously** across layers (chain rule through polynomials)
2. **Curvature is highly non-uniform** — some parameter directions are steep, others nearly flat
3. The shared `step_scale = loss / ||grad||²` is dominated by the steepest directions
4. Flat-direction parameters are starved, then suddenly overshoot when steep directions converge

This explains why the PLT training shows zig-zag with this optimizer but converges smoothly with AdamW (which uses per-parameter adaptive learning rates).

## Potential Fix Directions

If the coupling hypothesis is confirmed:

1. **Per-parameter scaling**: `p -= (loss / ||p.grad||²) * p.grad` for each parameter tensor independently
2. **Per-layer scaling**: compute `step_scale` per layer instead of globally
3. **Diagonal approximation**: use `loss * diag(H⁻¹) * grad` where H is an approximate Hessian
4. **Hybrid**: use the custom rule for direction, Adam-like scaling for magnitude
