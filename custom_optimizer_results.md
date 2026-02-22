# Custom Optimizer Experiment Results

## Update Rule Tested

`p -= (loss / ||grad||²) * grad`

This is Newton's root-finding method for `f(x) = 0`, not Newton's optimization method. All parameters share a single scalar `step_scale = loss / ||grad||²`.

## Results

### Experiment 1: Scalar `f(x) = (x-3)²`

Monotonic convergence, step_scale = 0.25 always, matches theory perfectly. The theoretical trajectory is `x_n = 3 + (x₀ - 3) / 2ⁿ` and the implementation reproduces it exactly. Float32 hits machine zero by step ~25, float64 continues to ~1e-23.

**Verdict**: No zig-zag. Single parameter means no coupling issue.

### Experiment 2: Symmetric `f(x,y) = (x-1)² + (y-2)²`

Also monotonic, step_scale = 0.25 always. Identical behavior to the scalar case — symmetry means the shared step_scale divides equally between both parameters.

**Verdict**: No zig-zag. Symmetric gradients mean no coupling issue.

### Experiment 3: Asymmetric `f(x,y) = (x-1)² + 100*(y-2)²` — KEY RESULT

**Zig-zag detected at 37 out of 100 steps.** Even on a pure quadratic with just 2 parameters. The loss jumps are severe:

| Step | Loss before | Loss after | Increase |
|------|------------|------------|----------|
| 5    | 7.91e+01   | 3.61e+02   | 4.6x     |
| 8    | 5.79e+01   | 3.56e+02   | 6.1x     |
| 13   | 2.68e+01   | 9.28e+01   | 3.5x     |

The mechanism: when y's gradient dominates `||grad||²`, step_scale is small, starving x. When y nears its target, its gradient shrinks, step_scale suddenly jumps, and y overshoots past its target. Then y's gradient grows large again, step_scale drops, and the cycle repeats.

The gradient ratio `|grad_x|/|grad_y|` oscillates between ~0.01 (y dominates) and ~1.2 (briefly balanced), confirming the alternating dominance pattern.

**Verdict**: Zig-zag confirmed on a simple quadratic. The shared step_scale is the root cause.

### Experiment 4: Linear Network `y = w*x + b`, fitting {(1,2), (2,5)}

**Zig-zag at 32 out of 100 steps.** The w-b coupling through the data matrix creates asymmetric gradients, triggering the same oscillation pattern. Example:

| Step | Loss before | Loss after |
|------|------------|------------|
| 4    | 4.19e-01   | 1.04e+00   |
| 8    | 2.74e-01   | 4.07e-01   |
| 15   | 3.67e-02   | 1.09e-01   |

Despite the oscillations, eventual convergence occurs — reaching loss ~5e-11 by step 90.

**Verdict**: Zig-zag present. Data-induced parameter coupling is sufficient to trigger it.

### Experiment 5: Nonlinear MLP with x² Activation

**Zig-zag at 54 steps (float32) and 78 steps (float64).** The non-convex landscape creates highly variable gradient magnitudes across the 13 parameters, amplifying the coupling problem. Example (float32):

| Step | Loss before | Loss after |
|------|------------|------------|
| 3    | 2.51e-01   | 3.86e-01   |
| 5    | 1.83e-01   | 3.62e-01   |
| 8    | 6.56e-02   | 1.72e-01   |

Despite heavy oscillation, both float32 and float64 eventually converge (float32 to ~2.4e-14, float64 to ~8.9e-32 by step 180).

**Verdict**: Zig-zag present and more pronounced. Non-convexity worsens the coupling but doesn't prevent eventual convergence.

## Key Takeaway

The zig-zag is **not** caused by non-convexity, floating point precision, or neural network complexity. It appears even on a simple 2-parameter quadratic (Experiment 3). The root cause is the **shared step_scale**: when parameter gradients have different magnitudes, the single scaling factor alternately overshoots different parameters.

| Experiment | Params | Convex? | Zig-zag? | Cause |
|-----------|--------|---------|----------|-------|
| 1. Scalar quadratic | 1 | Yes | No | No coupling possible |
| 2. Symmetric quadratic | 2 | Yes | No | Equal gradient magnitudes |
| 3. Asymmetric quadratic | 2 | Yes | **Yes** | Gradient magnitude imbalance |
| 4. Linear network | 2 | Yes | **Yes** | Data-induced coupling |
| 5. Nonlinear MLP | 13 | No | **Yes** | Coupling + non-convexity |

## Connection to PLT / debugLogicGates

The SymbolicLogicEngine's PLT attention creates polynomial mappings of degree up to 3^depth. With depth=3, outputs are degree-27 polynomials. This means:

1. Gradient magnitudes vary enormously across layers (chain rule through polynomials)
2. Curvature is highly non-uniform — some parameter directions are steep, others nearly flat
3. The shared `step_scale` is dominated by the steepest directions
4. Flat-direction parameters are starved, then suddenly overshoot when steep directions converge

This explains why PLT training shows zig-zag with this optimizer but converges smoothly with AdamW (which uses per-parameter adaptive learning rates).

## Potential Fixes

1. **Per-parameter scaling**: `p -= (loss / ||p.grad||²) * p.grad` for each parameter tensor independently
2. **Per-layer scaling**: compute step_scale per layer instead of globally
3. **Damping**: `p -= α * (loss / ||grad||²) * grad` with α < 1 to prevent overshooting
4. **Hybrid**: use the custom rule for direction, Adam-like scaling for magnitude
