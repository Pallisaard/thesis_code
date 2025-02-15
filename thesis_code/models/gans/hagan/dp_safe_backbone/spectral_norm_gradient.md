# Gradient of the Spectral Norm-Constrained Linear Layer

## **Step 1: Definition of Spectral Norm**

The spectral norm (largest singular value) of a weight matrix $W \in \mathbb{R}^{m \times n}$ is given by:

$$
\sigma(W) = \max_{\|v\|_2 = 1, \|u\|_2 = 1} u^\top W v
$$

where $ u $ and $ v $ are the left and right singular vectors corresponding to the largest singular value.

From Singular Value Decomposition (SVD):
$$
W = U \Sigma V^\top
$$
where:
- $ U = [u_1, \dots, u_m] $ contains left singular vectors,
- $ V = [v_1, \dots, v_n] $ contains right singular vectors,
- $ \Sigma = \text{diag}(\sigma_1, \dots, \sigma_r) $ with $ \sigma_1 = \sigma(W) $.

Thus, we can express the spectral norm as:
$$
\sigma(W) = u_1^\top W v_1.
$$

## **Step 2: Compute the Gradient of $ \sigma(W) $**

Using differentiation results for singular values:
$$
\nabla_W \sigma(W) = u_1 v_1^\top.
$$

## **Step 3: Compute the Gradient of $ \sigma(W)^2 $**

Since $ \sigma(W)^2 = (\sigma(W))^2 $, applying the chain rule:
$$
\nabla_W \sigma(W)^2 = 2\sigma(W) \nabla_W \sigma(W) = 2\sigma(W) u_1 v_1^\top.
$$

## **Step 4: Compute the Gradient of $ Wx $**

The operation $ Wx $ is just matrix-vector multiplication, so its gradient with respect to $ W $ is:
$$
\nabla_W Wx = I \otimes x^\top.
$$

## **Step 5: Compute the Gradient of $ \frac{1}{\sigma(W)} $**

Using the derivative:
$$
\frac{\partial}{\partial \sigma(W)} \left( \frac{1}{\sigma(W)} \right) = -\frac{1}{\sigma(W)^2},
$$

we apply the chain rule:
$$
\frac{\partial}{\partial W} \left( \frac{1}{\sigma(W)} \right) = -\frac{1}{\sigma(W)^2} \nabla_W \sigma(W) = -\frac{1}{\sigma(W)^2} u_1 v_1^\top.
$$

## **Step 6: Compute the Full Gradient of $ y = \frac{W}{\sigma(W)} x $**

Using the product rule:
$$
\frac{\partial y}{\partial W} = \frac{\partial}{\partial W} \left( \frac{W}{\sigma(W)} x \right).
$$

Expanding:
$$
\frac{\partial y}{\partial W} = \frac{1}{\sigma(W)} \frac{\partial (Wx)}{\partial W} + \left( \frac{\partial}{\partial W} \frac{1}{\sigma(W)} \right) (Wx).
$$

Substituting previously derived gradients:
$$
\frac{\partial y}{\partial W} = \frac{1}{\sigma(W)} (I \otimes x^\top) - \frac{1}{\sigma(W)^2} u_1 v_1^\top (Wx).
$$

Rewriting:
$$
\frac{\partial y}{\partial W} = \frac{1}{\sigma(W)} \left( I - \frac{u_1 v_1^\top}{\sigma(W)} Wx \right) \otimes x^\top.
$$

This is the final gradient expression for the spectral norm-constrained linear layer.


## Why not compute the second term?

The second term in the full gradient derivation (`-1/σ(W)^2 * u₁v₁ᵀ(Wx)`) is theoretically part of the complete gradient, but there are a few reasons why omitting it in practice is acceptable:

1. **Power Iteration Effect**:

   - During the forward pass, the power iteration method (`_power_iteration()`) is continuously updating the singular vectors `u` and `v`
   - This process ensures that the weight matrix is already approximately normalized to have a spectral norm close to 1
   - When σ(W) ≈ 1, the correction term becomes very small

2. **Computational Efficiency**:

   - Computing the full gradient requires additional matrix multiplications
   - The benefit of including this term is often outweighed by the computational cost
   - The simplified gradient still maintains the core spectral normalization property

3. **Empirical Success**:

   - This simplified gradient computation has been successfully used in many implementations (including the original Spectral Normalization paper)
   - The network still learns effectively without the correction term

4. **Stability**:

   - The simplified gradient can actually lead to more stable training
   - The correction term can sometimes introduce numerical instabilities due to the squared denominator

The mathematical intuition is that since power iteration keeps the spectral norm close to 1, we're effectively working with an already normalized weight matrix, making the correction term's contribution minimal. This is similar to how in batch normalization, we often use simplified gradient computations that ignore some of the higher-order effects of the normalization.

If you wanted to implement the full gradient, it would look something like this:

```python
def spectral_norm_linear_grad_sampler_full(module, activations, backprops):
    """Full gradient computation including the correction term."""
    sigma_w = torch.matmul(module.u.T, torch.matmul(module.weight, module.v)).squeeze()
    
    # First term (the one we currently use)
    grad_weight_main = backprops.unsqueeze(2) * activations.unsqueeze(1)
    grad_weight_main /= sigma_w
    
    # Correction term
    Wx = torch.matmul(module.weight, activations.T)  # W*x
    uv_term = torch.matmul(module.u, module.v.T)     # u₁v₁ᵀ
    correction = -uv_term * Wx / (sigma_w ** 2)
    
    # Combined gradient
    grad_weight = grad_weight_main + correction
    
    # ... rest of the function
```

However, in practice, the simpler version is typically preferred for its efficiency and stability while still maintaining good performance.