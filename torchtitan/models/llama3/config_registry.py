# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import (
    OptimizersContainer,
    OptimizersInBackwardContainer,
)
from torchtitan.components.quantization import BitLinearConverter, Float8LinearConverter
from torchtitan.components.validate import Validator
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import (
    ChatDataLoader,
    HuggingFaceTextDataLoader,
)
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from . import model_registry


def llama3_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(pipeline_parallel_schedule="Interleaved1F1B"),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        validator=Validator.Config(
            freq=5,
            steps=10,
        ),
    )


def llama3_debugmodel_fused_qkv() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_spec = model_registry("debugmodel_fused_qkv")
    return config


def llama3_debugmodel_flex_attn() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_spec = model_registry("debugmodel", attn_backend="flex")
    return config


def llama3_debugmodel_varlen_attn() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_spec = model_registry("debugmodel", attn_backend="varlen")
    return config


def llama3_debugmodel_opt_in_bwd() -> Trainer.Config:
    config = llama3_debugmodel()
    config.optimizer = OptimizersInBackwardContainer.Config(lr=8e-4)
    return config


def llama3_debugmodel_float8() -> Trainer.Config:
    config = llama3_debugmodel()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            Float8LinearConverter.Config(model_compile_enabled=model_compile_enabled),
        ],
    )
    return config


def llama3_debugmodel_float8_emulate_lora() -> Trainer.Config:
    from torchtitan.components.lora import LoRAConverter

    config = llama3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            Float8LinearConverter.Config(
                emulate=True,
                model_compile_enabled=False,
            ),
            LoRAConverter.Config(
                rank=8, alpha=16.0, target_modules=["wq", "wkv", "wo"]
            ),
        ],
    )
    return config


def llama3_debugmodel_ce_loss() -> Trainer.Config:
    """Debug model with standard (non-chunked) CrossEntropyLoss."""
    from torchtitan.components.loss import CrossEntropyLoss

    config = llama3_debugmodel()
    config.loss = CrossEntropyLoss.Config()
    return config


def llama3_160m() -> Trainer.Config:
    """~160M stock TorchTitan Llama-style training rung.

    This is the framework-sanity rung: same TorchTitan trainer, dataloader,
    metrics, activation checkpointing, and checkpointing path as the BitNet
    config, but with ordinary Linear layers.

    Seed-locked: ``debug.seed`` is fixed so model-init RNG is reproducible.
    ``llama3_160m_bitnet`` inherits this config, so the stock and BitNet runs
    initialize weights from the SAME seed. Combined with the dataloader's
    hard-coded shuffle/interleave seed (42), both variants see an identical
    weight init AND identical data order -- so their step-0 losses match and
    any later divergence is attributable to quantization, not RNG. Override on
    the CLI with ``--debug.seed N`` to run additional seeds.
    """

    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("160M"),
        debug=DebugConfig(seed=42),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=20,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=512,
            steps=100,
            dtype="bfloat16",
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=50,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        validator=Validator.Config(freq=50, steps=10),
    )


def llama3_160m_bitnet() -> Trainer.Config:
    """~160M BitNet b1.58-style TorchTitan training rung.

    This config uses the same model shape and training settings as
    ``llama3_160m`` but swaps internal Linear layers to BitLinear through the
    converter mechanism. The lm_head/output projection is intentionally left as
    a normal Linear by the converter's default filter.
    """

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


def _llama3_160m_bitnet_ablation(
    *, activation_quant: bool, weight_quant: bool
) -> Trainer.Config:
    """Shared builder for the gap-attribution ablation configs.

    Identical to ``llama3_160m_bitnet`` (same seed lock, same model shape, same
    training settings, same lm_head filter, same ``pre_norm=True`` structure)
    except for which quantizers run inside BitLinear's forward. See
    ``docs/bitnet-160m-mvp/experiments/2026-06-12-gap-attribution/``.
    """

    config = llama3_160m()
    config.model_spec = model_registry(
        "160M",
        converters=[
            BitLinearConverter.Config(
                filter_fqns=["output", "lm_head"],
                activation_quant=activation_quant,
                weight_quant=weight_quant,
                pre_norm=True,
            )
        ],
    )
    return config


def llama3_160m_bitnet_fp16_weights() -> Trainer.Config:
    """Ablation: BitLinear structure + int8 activation quant, NO ternary weights.

    Weights stay full-precision in the forward pass (no absmean ternarization,
    no weight STE). Isolates the contribution of weight ternarization: the gap
    between this and full BitNet is what ternary weights cost.
    """

    return _llama3_160m_bitnet_ablation(activation_quant=True, weight_quant=False)


def llama3_160m_bitnet_no_actquant() -> Trainer.Config:
    """Ablation: BitLinear structure + ternary weights, NO activation quant.

    Activations flow through unquantized (no per-token int8 absmax, no
    activation STE). Isolates the contribution of activation quantization: the
    gap between this and full BitNet is what int8 activations cost.
    """

    return _llama3_160m_bitnet_ablation(activation_quant=False, weight_quant=True)


def llama3_160m_bitnet_structure_only() -> Trainer.Config:
    """Ablation: BitLinear structure only — no weight or activation quant.

    BitLinear modules with their extra pre-RMSNorm (SubLN-style) are in place,
    but the forward math is otherwise a plain F.linear. Isolates the purely
    structural/optimization effect of the extra norms; expected to land at or
    near stock. Note the head is stock in ALL these configs (filter_fqns), so
    a separate "fp16 head" variant is redundant — that is already the baseline.
    """

    return _llama3_160m_bitnet_ablation(activation_quant=False, weight_quant=False)


def llama3_8b() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./assets/hf/Llama-3.1-8B",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_registry("8B"),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def llama3_70b() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./assets/hf/Llama-3.1-70B",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_registry("70B"),
        optimizer=OptimizersContainer.Config(lr=1.5e-4),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=8192,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def llama3_405b() -> Trainer.Config:
    compile_config = CompileConfig(enable=True)
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./assets/hf/Llama-3.1-405B",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_registry(
            "405B",
            converters=[
                Float8LinearConverter.Config(
                    filter_fqns=["output"],
                    model_compile_enabled=(
                        compile_config.enable and "model" in compile_config.components
                    ),
                ),
            ],
        ),
        optimizer=OptimizersContainer.Config(lr=8e-5),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=8192,
            steps=3000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=8,
            enable_async_tensor_parallel=True,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=compile_config,
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def sft_debugmodel() -> Trainer.Config:
    """SFT debug config with Llama3 debugmodel and local test data."""

    def process_sample(sample):
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    model_spec = model_registry("debugmodel", attn_backend="flex")

    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_spec,
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        dataloader=ChatDataLoader.Config(
            dataset_path="json",
            load_dataset_kwargs={
                "data_files": "tests/assets/sft_test/data.json",
                "split": "train",
            },
            sample_processor=process_sample,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )
