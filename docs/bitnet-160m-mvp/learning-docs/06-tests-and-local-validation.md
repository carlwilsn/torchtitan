# 05 — Testing strategy and monitored structures

Testing happened in layers. The point was to avoid spending GPU time on bugs that can be caught by small local tests.

## Test ladder

The validation ladder was:

1. Unit tests for quantization math and gradients.
2. Python compile checks for edited files.
3. GPU unit tests in the real Linux/PyTorch environment.
4. Stock TorchTitan debug smoke.
5. Stock 160M TorchTitan smoke.
6. BitNet 160M TorchTitan smoke.

This order matters. You should not start with the BitNet 160M run because a failure there has too many possible causes.

## Unit tests

File:

```text
tests/unit_tests/test_bitnet_quantization.py
```

### Test 1: `test_weight_quant_forward_values_are_scaled_ternary`

Purpose:

- Verify the forward output of `weight_quant` only contains scaled ternary values.
- Verify gradients flow to the original weight tensor.

Important code:

```python
w = torch.tensor([[-2.0, -0.2, 0.2, 2.0]], requires_grad=True)
q = weight_quant(w)
gamma = w.detach().abs().mean()
allowed = torch.tensor([-gamma, 0.0, gamma], dtype=q.dtype)
assert all(torch.any(torch.isclose(v, allowed)) for v in q.flatten())

q.sum().backward()
assert w.grad is not None
assert torch.all(w.grad != 0)
```

What is monitored:

- `q.flatten()` values.
- `w.grad` after backward.

Why this catches bugs:

- If quantization produces non-ternary values, the BitNet weight path is wrong.
- If gradients are zero or missing, the STE is wrong.

### Test 2: `test_activation_quant_preserves_shape_dtype_and_grad`

Purpose:

- Verify activation quantization does not change tensor shape or dtype.
- Verify gradients flow through activation quantization.

Important code:

```python
x = torch.randn(2, 3, 8, dtype=torch.float32, requires_grad=True)
y = activation_quant(x)
assert y.shape == x.shape
assert y.dtype == x.dtype

y.sum().backward()
assert x.grad is not None
assert torch.all(x.grad != 0)
```

What is monitored:

- Shape.
- Dtype.
- `x.grad`.

Why this matters:

- Transformer modules expect exact shape preservation.
- Dtype changes can break mixed precision or cause unexpected memory/performance behavior.
- Gradients must flow through quantized activations.

### Test 3: `test_bitlinear_forward_backward_with_prenorm`

Purpose:

- Verify `BitLinear` can be constructed, initialized, run forward, and backpropagated.
- Verify the optional pre-norm participates in training.

Important code:

```python
layer = BitLinear(BitLinear.Config(in_features=8, out_features=4, bias=False, pre_norm=True))
layer.init_states()

x = torch.randn(2, 3, 8, requires_grad=True)
y = layer(x)
assert y.shape == (2, 3, 4)

y.sum().backward()
assert layer.weight.grad is not None
assert layer.pre_norm is not None
assert layer.pre_norm.weight.grad is not None
```

What is monitored:

- Output shape.
- `layer.weight.grad`.
- `layer.pre_norm.weight.grad`.

Why this matters:

- It proves the layer is trainable, not just callable.
- It proves the RMSNorm path is connected to the graph.

### Test 4: `test_bitlinear_converter_replaces_matching_linear_configs`

Purpose:

- Verify the converter mutates config trees correctly.
- Verify filters prevent conversion where requested.
- Verify list-contained configs are handled.

The test defines a tiny config tree:

```python
class TinyConfigTree(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        keep: Linear.Config
        swap: Linear.Config
        nested: list[Linear.Config]
```

Then it builds:

```python
tree = TinyConfigTree.Config(
    keep=Linear.Config(in_features=8, out_features=8),
    swap=Linear.Config(in_features=8, out_features=8),
    nested=[Linear.Config(in_features=8, out_features=8)],
)
```

Conversion:

```python
BitLinearConverter(BitLinearConverter.Config(filter_fqns=["keep"])).convert(tree)
```

Assertions:

```python
assert isinstance(tree.keep, Linear.Config)
assert not isinstance(tree.keep, BitLinear.Config)
assert isinstance(tree.swap, BitLinear.Config)
assert isinstance(tree.nested[0], BitLinear.Config)
```

What is monitored:

- Type of `tree.keep`.
- Type of `tree.swap`.
- Type of `tree.nested[0]`.

Why this matters:

- TorchTitan models are config trees, not simple flat module lists.
- If list replacement is wrong, layers inside lists of transformer blocks could fail to convert.

## Compile checks

Local command:

```bash
python -m py_compile \
  torchtitan/components/quantization/bitnet.py \
  torchtitan/components/quantization/__init__.py \
  torchtitan/components/quantization/utils.py \
  torchtitan/models/llama3/__init__.py \
  torchtitan/models/llama3/config_registry.py \
  tests/unit_tests/test_bitnet_quantization.py
```

Purpose:

- Catch syntax errors across all edited Python files.
- This is cheap and should run before GPU work.

## GPU smoke tests

The GPU tests monitored different structures than unit tests.

They monitored:

- whether TorchTitan imports in Linux environment,
- whether CUDA is visible,
- whether the trainer can instantiate model/loss/optimizer/dataloader,
- parameter counts,
- converter log count,
- final step/loss log line,
- whether backward and optimizer step complete.

## Why use stock baselines before BitNet?

A BitNet training failure could be caused by:

- broken GPU environment,
- incompatible PyTorch/TorchTitan versions,
- dataloader issue,
- trainer bug,
- FSDP issue,
- model architecture issue,
- BitNet layer issue.

Running stock debug and stock 160M first removes many variables.

If stock fails, do not blame BitNet. That happened during validation: stock failed first in the one-GPU FSDP path.

## Invariants to preserve in future edits

When changing the implementation, preserve these invariants:

1. `activation_quant(x)` returns same shape and dtype as `x`.
2. `weight_quant(w)` returns same shape and dtype as `w`.
3. `weight_quant` forward values are scaled ternary.
4. Both quantizers allow nonzero gradients to the original tensors.
5. `BitLinear` owns a full-precision latent `weight` parameter.
6. `BitLinear.Config` copies linear structural fields from `Linear.Config`.
7. Converter does not quantize `lm_head` unless intentionally changed.
8. Stock 160M config remains available for baseline comparison.
9. BitNet config should differ from stock primarily through `model_spec` conversion.
10. Any GPU result should include the exact command, environment, and final log line.

## What still needs better tests

Missing tests to add later:

- Count expected converted layers for the real 160M config.
- Verify `lm_head` remains unconverted in the real model config.
- Verify parameter initialization is identical between stock and BitNet latent weights where shapes match.
- Verify disabling `activation_quant`, `weight_quant`, or `pre_norm` changes behavior as expected.
- Add a tiny trainer-level test if TorchTitan supports one in CI.
- Add checkpoint save/load test for `BitLinear`.
