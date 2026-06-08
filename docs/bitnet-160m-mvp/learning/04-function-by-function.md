# 04 — Function-by-function implementation walkthrough

This page walks through the implemented files at the level of functions/classes.

## `torchtitan/components/quantization/bitnet.py`

### `_EPS = 1e-5`

Small constant used to avoid division by zero in quantization scales.

Why needed:

- If all activations or weights are zero, `amax` or `mean(abs())` can be zero.
- Dividing by zero would create `inf` or `nan`.
- `clamp_min(eps)` keeps the denominator safe.

### `_ste_round(x)`

```python
def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()
```

Purpose:

- Implements round in the forward pass while making backward behave like identity.

Current status:

- It exists but the main quantization functions inline the broader STE pattern directly.
- It can be used later to simplify code, but is not essential.

### `activation_quant(x, eps=_EPS)`

Purpose:

- Fake-quantize activations to int8-like values per token/vector.

Input:

- `x`: activation tensor, typically shaped `[batch, seq, hidden]`.

Important data structures:

- `scale`: tensor shaped like `[batch, seq, 1]` when `x` is `[batch, seq, hidden]`.
- `y`: quantized/dequantized activation tensor same shape and dtype as `x`.

Algorithm:

1. Compute max absolute value across the last dimension:

   ```python
   x.abs().amax(dim=-1, keepdim=True)
   ```

2. Convert that to a scale mapping the largest magnitude to 127:

   ```python
   scale = 127.0 / ...
   ```

3. Multiply by scale, round, clamp, then divide by scale:

   ```python
   y = torch.round(x * scale).clamp(-128, 127) / scale
   ```

4. Return an STE value:

   ```python
   return x + (y - x).detach()
   ```

Why per-token scale:

- Each token’s hidden vector can have a different magnitude.
- Per-token scaling preserves more relative precision than one global activation scale.
- It matches common LLM activation quantization practice.

### `weight_quant(w, eps=_EPS)`

Purpose:

- Fake-quantize weights to BitNet ternary values.

Input:

- `w`: full-precision latent weight matrix, usually `[out_features, in_features]`.

Important data structures:

- `gamma`: scalar tensor, the absmean scale.
- `y`: ternary scaled weight matrix same shape and dtype as `w`.

Algorithm:

1. Compute absmean scale:

   ```python
   gamma = w.abs().mean().clamp_min(eps)
   ```

2. Normalize weights by `gamma`, round, and clamp to ternary range:

   ```python
   torch.round(w / gamma).clamp(-1, 1)
   ```

3. Rescale by `gamma`:

   ```python
   y = ... * gamma
   ```

4. Return STE value:

   ```python
   return w + (y - w).detach()
   ```

Why one absmean scale:

- It is simple and close to BitNet-style ternary scaling.
- It avoids adding many per-channel scale tensors in the MVP.
- It keeps the first implementation easy to inspect and test.

### `class BitLinear(Module)`

Purpose:

- Drop-in TorchTitan-compatible replacement for `Linear`.

Why subclass `Module`:

- TorchTitan modules follow a protocol/config pattern.
- `Module` provides expected initialization behavior such as `init_states()`.

#### `BitLinear.Config`

```python
@dataclass(kw_only=True, slots=True)
class Config(Linear.Config):
    activation_quant: bool = True
    weight_quant: bool = True
    pre_norm: bool = True
    eps: float = _EPS
```

Purpose:

- Store construction arguments for `BitLinear`.
- Inherit standard linear config fields.

Inherited fields include:

- `in_features`,
- `out_features`,
- `bias`,
- `param_init`,
- `sharding_config`.

New fields:

- `activation_quant`: enable/disable activation fake quantization.
- `weight_quant`: enable/disable weight fake quantization.
- `pre_norm`: enable/disable RMSNorm before quantization.
- `eps`: numerical safety epsilon.

Why make toggles:

- For ablations.
- For debugging whether failures come from activation quantization, weight quantization, or pre-norm.
- For future comparison against paper variants.

#### `BitLinear.__init__(self, config)`

Constructs trainable state:

- `self.weight`: full-precision latent weight parameter.
- `self.bias`: optional bias parameter or registered `None`.
- `self.pre_norm`: optional RMSNorm module.

Important: the trainable `weight` is not ternary storage. It is the latent weight that receives gradients.

Why copy linear attributes:

- `in_features` and `out_features` mirror `nn.Linear`/TorchTitan `Linear` behavior.
- Other TorchTitan utilities may inspect these attributes.

#### `BitLinear.reset_parameters()`

Fallback initializer.

Why it exists:

- If TorchTitan’s normal param-init path does not provide initialization, this layer still initializes safely like `nn.Linear`.

Normal path:

- In Llama configs, `param_init` dictionaries are copied into `BitLinear.Config`, so TorchTitan’s `Module.init_states()` should apply the intended initialization.

#### `BitLinear.forward(x)`

```python
if self.pre_norm is not None:
    x = self.pre_norm(x)
if self.activation_quant_enabled:
    x = activation_quant(x, eps=self.eps)

w = self.weight
if self.weight_quant_enabled:
    w = weight_quant(w, eps=self.eps)

return F.linear(x, w, self.bias)
```

Data flow:

1. Input activations enter as full-precision/bfloat16 tensors.
2. Optional RMSNorm stabilizes them.
3. Activation quantization produces int8-like dequantized activations.
4. Weight quantization produces ternary-like dequantized weights.
5. `F.linear` computes output using ordinary PyTorch matmul.

Why use `F.linear`:

- It keeps autograd, dtype handling, and GPU kernels simple for MVP.
- It avoids writing custom kernels before proving integration.

### `class BitLinearConverter(QuantizationConverter)`

Purpose:

- Mutate a model config tree before module construction.
- Replace selected `Linear.Config` nodes with `BitLinear.Config` nodes.

#### `BitLinearConverter.Config`

Fields:

- `filter_fqns`: substrings to skip, default `['output', 'lm_head']`.
- `activation_quant`: passed to each created `BitLinear.Config`.
- `weight_quant`: passed to each created `BitLinear.Config`.
- `pre_norm`: passed to each created `BitLinear.Config`.
- `eps`: passed to each created `BitLinear.Config`.
- `require_dim_multiple_of_16`: optionally reuse float8-style hardware filter.

Why `filter_fqns`:

- Some modules should stay unquantized.
- The output head is intentionally skipped for stability.

Why optional dim multiple of 16:

- Float8 tensorcore paths need dimension multiples of 16.
- BitNet fake quantization does not require this, so default is false.
- The option remains useful if future kernels impose constraints.

#### `BitLinearConverter.__init__`

Builds `self.filter_fn`.

If `require_dim_multiple_of_16=True`, uses TorchTitan’s existing `module_filter_fn`.

Otherwise, uses a simple FQN substring skip:

```python
lambda _linear_config, fqn: not any(filter_fqn in fqn for filter_fqn in config.filter_fqns)
```

FQN means fully qualified name: the path to a config inside the nested model config tree.

#### `BitLinearConverter.convert(model_config)`

Core loop:

```python
for fqn, linear_config, parent, attr in model_config.traverse(Linear.Config):
    ...
```

Important data structures returned by traversal:

- `fqn`: string path identifying where the config was found.
- `linear_config`: the existing `Linear.Config` object.
- `parent`: object or list that owns the config.
- `attr`: attribute name or list index where the config lives.

For each matching linear config:

1. Skip if already `BitLinear.Config`.
2. Skip if filter rejects it.
3. Create a new `BitLinear.Config` copying structural fields.
4. Replace the old config in its parent.
5. Increment `converted`.

Why copy `param_init` and `sharding_config`:

- Parameter initialization should remain the same as the stock Llama layer.
- Distributed/sharding metadata should survive conversion.

## `torchtitan/components/quantization/utils.py`

### `has_quantization(model_config)`

Purpose:

- Detect whether a config tree contains quantized modules.

BitNet change:

```python
quant_linear_types = [BitLinear.Config, MXFP8Linear.Config]
```

Why this matters:

- A `BitLinear.Config` is a kind of `Linear.Config`, but TorchTitan needs to know it represents quantization.
- Without this, some quantization-aware code paths may not activate.

## `torchtitan/models/llama3/__init__.py`

### `_160m(attn_backend)`

Purpose:

- Define a small Llama-style architecture for the MVP.

Key choices:

```python
dim = 1024
n_heads = 16
n_kv_heads = 4
n_layers = 14
vocab_size = 2048
```

Why these values:

- `dim=1024` and `14` layers make the transformer body nontrivial but still A10-smoke-testable.
- `n_heads=16` gives head dimension 64.
- `n_kv_heads=4` uses grouped-query attention, closer to modern Llama shapes.
- `vocab_size=2048` uses the local test tokenizer so no gated Llama assets are required.

Important caveat:

- Real Llama vocab is much larger. With a 128k vocab, embedding/output parameters dominate and the parameter count changes substantially.

### `llama3_configs`

Added:

```python
"160M": _160m
```

This registers the new model flavor.

### `model_registry(flavor, attn_backend='sdpa', converters=None)`

Existing function used by the new config.

Key behavior:

```python
config = llama3_configs[flavor](attn_backend=attn_backend)
if converters is not None:
    validate_converter_order(converters)
    for c in converters:
        c.build().convert(config)
```

This is exactly where the BitNet converter mutates the model config.

## `torchtitan/models/llama3/config_registry.py`

### `llama3_160m()`

Purpose:

- Stock baseline training recipe for the 160M model.

Important fields:

- `loss=ChunkedCELoss.Config()`
- `hf_assets_path='./tests/assets/tokenizer'`
- `model_spec=model_registry('160M')`
- `optimizer=OptimizersContainer.Config(lr=3e-4)`
- scheduler warmup/decay settings
- `training=TrainingConfig(local_batch_size=2, seq_len=512, steps=100, dtype='bfloat16')`
- `dataloader=HuggingFaceTextDataLoader.Config(dataset='c4_test')`
- metrics log every step
- checkpoint every 50 steps
- selective activation checkpointing
- validation every 50 steps

Why stock baseline matters:

- If stock 160M fails, BitNet failure is not meaningful.
- Always establish the framework baseline first.

### `llama3_160m_bitnet()`

Purpose:

- Same training recipe as stock 160M, but with model config conversion.

Key code:

```python
config = llama3_160m()
config.model_spec = model_registry(
    "160M",
    converters=[
        BitLinearConverter.Config(
            filter_fqns=["output", "lm_head"],
            activation_quant=True,
            weight_quant=True,
            pre_norm=True,
        )
    ],
)
return config
```

Why reuse `llama3_160m()`:

- Keeps optimizer, scheduler, dataloader, metrics, and training settings identical.
- Makes stock-vs-BitNet comparisons cleaner.

Why replace only `model_spec`:

- The architecture shape is the same.
- Only internal linear module type changes.

## `tests/unit_tests/test_bitnet_quantization.py`

Covered in detail in the testing doc, but the file verifies:

- ternary weight values,
- gradient flow through weight quantization,
- activation shape/dtype/gradient preservation,
- BitLinear forward/backward,
- converter replacement behavior.
