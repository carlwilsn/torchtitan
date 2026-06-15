import torch

# Local torch is newer than torchtitan pins; shim the one missing FSDP symbol
# that torchtitan.models.llama3.parallelize imports, so we can build the model
# config and count params. This does NOT touch the model math at all.
import torch.distributed.fsdp as _fsdp

for _name in ("DataParallelMeshDims", "FSDPMeshDims", "HSDPMeshDims"):
    if not hasattr(_fsdp, _name):
        setattr(_fsdp, _name, type(_name, (), {}))

from torchtitan.models.llama3 import model_registry  # noqa: E402


def build(flavor):
    spec = model_registry(flavor)
    with torch.device("cpu"):
        model = spec.model.build()
    model.init_states(buffer_device=torch.device("cpu"))
    return model


for flavor, target in [("50M", 50e6), ("160M", 158e6), ("400M", 400e6)]:
    m = build(flavor)
    total = sum(p.numel() for p in m.parameters())
    emb = m.tok_embeddings.weight.numel()
    body = total - emb  # weight tying => lm_head shares embedding, counted once
    dev = (body - target) / target * 100
    print(
        f"{flavor}: total={total:,}  emb={emb:,}  body(non-emb)={body:,}  "
        f"target={target/1e6:.0f}M  dev={dev:+.1f}%"
    )
