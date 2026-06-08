# 02 — BitNet b1.58 mechanics implemented here

## What BitNet b1.58 means at this stage

BitNet b1.58 represents weights using ternary values:

$$
W_q \in \{-\gamma, 0, +\gamma\}
$$

The name “1.58-bit” comes from ternary storage having three possible states. Since:

$$
\log_2(3) \approx 1.585
$$

three states require about 1.58 bits of information per weight in an ideal packed representation.

This MVP does **training-time fake quantization**, not packed inference. That means:

- The trainable parameter remains a full-precision latent weight tensor.
- The forward pass uses quantized/dequantized values.
- The backward pass uses a straight-through estimator so gradients still update the full-precision latent weights.
- Optimizer state is still full precision / normal PyTorch optimizer state.

## Why not pack weights immediately?

Packing ternary weights is useful for deployment and inference kernels. It is not the first thing needed for training integration.

For training, we need:

1. A differentiable-ish training path.
2. Compatibility with TorchTitan’s module/config system.
3. Compatibility with autograd, optimizer, FSDP, metrics, and checkpointing.

A packed representation would complicate optimizer updates because the optimizer needs latent continuous weights, not only the ternary values.

## Activation quantization

Implemented function:

```python
def activation_quant(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    y = torch.round(x * scale).clamp(-128, 127) / scale
    y = y.to(dtype=x.dtype)
    return x + (y - x).detach()
```

Conceptually:

1. Treat the last dimension as the hidden/channel dimension.
2. For each token vector, compute the maximum absolute activation.
3. Choose a scale that maps that maximum to approximately int8 range.
4. Round to integer-like values.
5. Clamp to signed int8 range.
6. Dequantize back to the original dtype so PyTorch can use ordinary matrix multiply.
7. Use STE so gradients flow as if the quantizer were identity.

The scale is:

$$
scale = \frac{127}{\max(|x|, \epsilon)}
$$

The forward quantized/dequantized value is roughly:

$$
y = \frac{\mathrm{clamp}(\mathrm{round}(x \cdot scale), -128, 127)}{scale}
$$

## Weight quantization

Implemented function:

```python
def weight_quant(w: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    gamma = w.abs().mean().clamp_min(eps)
    y = torch.round(w / gamma).clamp(-1, 1) * gamma
    y = y.to(dtype=w.dtype)
    return w + (y - w).detach()
```

Conceptually:

1. Compute one global absmean scale for the weight matrix.
2. Divide weights by that scale.
3. Round to nearest integer.
4. Clamp to `-1`, `0`, or `+1`.
5. Multiply back by the scale.
6. Use STE so gradients update the original full-precision weights.

The scale is:

$$
\gamma = \mathrm{mean}(|W|)
$$

The forward ternary value is:

$$
W_q = \gamma \cdot \mathrm{clamp}(\mathrm{round}(W / \gamma), -1, 1)
$$

## Straight-through estimator

The core STE pattern used in both activation and weight quantization is:

```python
return x + (y - x).detach()
```

Forward:

- `(y - x).detach()` has value `y - x`.
- So the returned value is `x + y - x = y`.

Backward:

- `.detach()` blocks gradients through `y - x`.
- The gradient sees only `x`.
- So the quantizer behaves like identity for gradients.

This is important because `round` and `clamp` have zero or undefined gradients in the regions we care about. Without STE, training would not send useful gradients to the latent weights or activations.

## Pre-normalization inside BitLinear

`BitLinear` optionally applies RMSNorm before activation quantization:

```python
self.pre_norm = RMSNorm(...) if config.pre_norm else None
```

This is based on BitNet-style SubLN / normalization-before-quantization intuition: quantizers behave better when the activation distribution is stabilized.

In this MVP:

- `pre_norm=True` by default.
- The RMSNorm has its own trainable weight.
- This increases parameter count compared to stock `Linear` layers.

That increase is one reason the BitNet model parameter count is slightly larger than stock in the smoke logs.

## What is intentionally not implemented

This MVP does not yet implement:

- custom ternary CUDA kernels,
- packed 1.58-bit storage,
- inference-time dequantization kernels,
- optimizer-state compression,
- paper-perfect hyperparameter schedule,
- large-scale validation curves.

It implements the minimum correct training-stack integration: fake-quantized forward with STE backward inside TorchTitan.
