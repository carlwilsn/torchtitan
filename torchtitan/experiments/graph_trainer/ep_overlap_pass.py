# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Schedule already chunked EP token-exchange regions.

Contract
========
This pass is intentionally a scheduler only.  It consumes a graph that has
already been chunked by either eager chunking or ``ep_chunk_pass`` and must not
change tensor values, live-in/live-out materialization, or provenance.  The only
semantic input it relies on is chunk-body metadata collected by
``collect_chunked_regions``.

For each selected forward/backward region:

* exactly two chunk bodies, chunk 0 and chunk 1, must be present;
* true EP scheduling markers are
  ``_c10d_functional.all_to_all_single.default`` launches inside the selected
  chunk body;
* ``custom[_EP_TOKEN_EXCHANGE]`` is optional provenance/debug labeling only;
  generic traceback annotations on non-all-to-all nodes are sanitized and are
  not scheduling markers;
* marker counts must match across chunks;
* forward emits marker pairs in chunk order 0 then 1, backward emits 1 then 0;
* after each marker pair, ready non-collective body work is emitted as filler
  before advancing to the next marker pair;
* all graph nodes are emitted exactly once and the final graph must lint.

The same contract covers eager and graph chunking.  If a chunked region violates
the contract, the pass errors rather than producing a silent schedule change.

Pseudo-code
===========
1. Collect chunked regions and build node -> chunk-owner lookup from shared EP
   pass metadata.
2. For each region, collect true token-exchange all-to-all markers per chunk
   and sanitize generic EP annotations from non-marker nodes.
3. Validate both chunks have the same marker signature, then build dependency
   closures needed to launch each marker.
4. Emit wait-gated phases: marker pair in chunk order, ready filler work that
   does not need token-exchange waits, then final tail work with waits allowed.
5. Apply the requested region phases through a global topological emitter, lint,
   recompile, and validate that phase order materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx

from torchtitan.experiments.graph_trainer.common_utils import (
    _EP_TOKEN_EXCHANGE,
    _EP_TOKEN_EXCHANGE_WAIT,
)
from torchtitan.experiments.graph_trainer.ep_pass_utils import (
    ChunkBody,
    ChunkedRegion,
    ChunkOwner,
    collect_chunked_regions,
    ordered_nodes,
)
from torchtitan.tools.logging import logger


_GRAPH_BOUNDARY_OPS = {"placeholder", "get_attr"}
_EP_PHASES = {"dispatch", "combine"}


# Step 0: Small metadata helpers and local scheduling records.


@dataclass(frozen=True)
class _TokenExchange:
    label: str
    launch: fx.Node


@dataclass(frozen=True)
class _ScheduledRegion:
    region: ChunkedRegion
    phases: tuple[tuple[fx.Node, ...], ...]


def _custom_meta(node: fx.Node) -> dict[str, Any]:
    """Return mutable custom metadata when present, otherwise an empty dict."""
    custom = node.meta.get("custom")
    return custom if isinstance(custom, dict) else {}


def _ep_label(node: fx.Node) -> str:
    """Return the optional EP phase label for logs/wait metadata."""
    phase = _custom_meta(node).get(_EP_TOKEN_EXCHANGE)
    return phase if phase in _EP_PHASES else "all_to_all"


def _is_token_exchange_launch(node: fx.Node) -> bool:
    """Return whether a node is the true all-to-all launch scheduling marker."""
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.all_to_all_single.default
    )


def _is_c10d_functional_node(node: fx.Node) -> bool:
    """Return whether a node is a distributed functional op."""
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace == "_c10d_functional"
    )


def _same_region_owner(
    node: fx.Node,
    *,
    owner_by_node: dict[fx.Node, ChunkOwner],
    root_fqn: str,
    is_backward: bool,
) -> ChunkOwner | None:
    """Return chunk ownership only when it belongs to the same selected region."""
    owner = owner_by_node.get(node)
    if (
        owner is not None
        and owner.root_fqn == root_fqn
        and owner.is_backward == is_backward
    ):
        return owner
    return None


def _collect_token_exchanges(body: ChunkBody) -> tuple[_TokenExchange, ...]:
    """Step 2: collect true token-exchange launches for one chunk body."""
    node_set = set(body.nodes)
    exchanges: list[_TokenExchange] = []
    for node in body.nodes:
        if not _is_token_exchange_launch(node):
            phase = _custom_meta(node).get(_EP_TOKEN_EXCHANGE)
            if phase is None:
                continue
            logger.debug(
                "ep_overlap sanitized non-marker EP annotation: node=%s phase=%s",
                node.name,
                phase,
            )
            custom = dict(_custom_meta(node))
            custom.pop(_EP_TOKEN_EXCHANGE, None)
            node.meta["custom"] = custom
            continue

        label = _ep_label(node)
        waits = [
            user
            for user in node.users
            if user in node_set
            and user.op == "call_function"
            and user.target == torch.ops._c10d_functional.wait_tensor.default
        ]
        if len(waits) != 1:
            raise ValueError(
                f"ep_overlap expected one token-exchange wait for {node.name}, "
                f"found {len(waits)}."
            )
        wait = waits[0]
        custom = dict(_custom_meta(wait))
        custom.pop(_EP_TOKEN_EXCHANGE, None)
        custom[_EP_TOKEN_EXCHANGE_WAIT] = label
        wait.meta["custom"] = custom
        exchanges.append(_TokenExchange(label=label, launch=node))
    return tuple(exchanges)


def _exchange_signature(exchanges: tuple[_TokenExchange, ...]) -> tuple[str, ...]:
    """Return the semantic marker sequence used to match chunk pairs."""
    return ("all_to_all",) * len(exchanges)


def _exchange_labels(exchanges: tuple[_TokenExchange, ...]) -> tuple[str, ...]:
    """Return optional marker labels for diagnostics."""
    return tuple(exchange.label for exchange in exchanges)


# Step 3: Build marker dependency closures and identify ready filler work.


def _hidden_body_deps(
    node: fx.Node,
    *,
    owner_by_node: dict[fx.Node, ChunkOwner],
    root_fqn: str,
    is_backward: bool,
) -> tuple[fx.Node, ...]:
    """Find same-region body deps behind unowned graph plumbing."""
    deps: list[fx.Node] = []
    seen: set[fx.Node] = set()
    stack = list(node.all_input_nodes)
    while stack:
        dep = stack.pop()
        if dep in seen:
            continue
        seen.add(dep)
        owner = _same_region_owner(
            dep,
            owner_by_node=owner_by_node,
            root_fqn=root_fqn,
            is_backward=is_backward,
        )
        if owner is not None:
            deps.append(dep)
        elif owner_by_node.get(dep) is None and dep.op not in _GRAPH_BOUNDARY_OPS:
            stack.extend(dep.all_input_nodes)
    return tuple(deps)


def _body_deps(
    node: fx.Node,
    *,
    body: ChunkBody,
    owner_by_node: dict[fx.Node, ChunkOwner],
) -> tuple[fx.Node, ...]:
    """Return same-region body deps, including deps behind unowned plumbing."""
    deps: list[fx.Node] = []
    for dep in node.all_input_nodes:
        owner = _same_region_owner(
            dep,
            owner_by_node=owner_by_node,
            root_fqn=body.owner.root_fqn,
            is_backward=body.owner.is_backward,
        )
        if owner is not None:
            deps.append(dep)
        elif owner_by_node.get(dep) is None and dep.op not in _GRAPH_BOUNDARY_OPS:
            deps.extend(
                _hidden_body_deps(
                    dep,
                    owner_by_node=owner_by_node,
                    root_fqn=body.owner.root_fqn,
                    is_backward=body.owner.is_backward,
                )
            )
    return tuple(dict.fromkeys(deps))


def _marker_closure(
    launch: fx.Node,
    *,
    body: ChunkBody,
    order: dict[fx.Node, int],
    owner_by_node: dict[fx.Node, ChunkOwner],
    exchange_index: int,
    exchange_indices: dict[fx.Node, int],
) -> tuple[fx.Node, ...]:
    """Return the body nodes required to launch one token exchange."""
    closure: list[fx.Node] = []
    visiting: set[fx.Node] = set()
    visited: set[fx.Node] = set()

    def visit(node: fx.Node, *, allow_peer_chunk: bool = False) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ValueError(
                "ep_overlap found a cycle while building marker closure for "
                f"{launch.name} in {body.owner}."
            )

        owner = _same_region_owner(
            node,
            owner_by_node=owner_by_node,
            root_fqn=body.owner.root_fqn,
            is_backward=body.owner.is_backward,
        )
        if owner is None:
            for dep in sorted(
                _hidden_body_deps(
                    node,
                    owner_by_node=owner_by_node,
                    root_fqn=body.owner.root_fqn,
                    is_backward=body.owner.is_backward,
                ),
                key=order.__getitem__,
            ):
                visit(dep, allow_peer_chunk=True)
            return

        if owner.chunk_id != body.owner.chunk_id and not allow_peer_chunk:
            raise ValueError(
                "ep_overlap cannot schedule a token exchange whose dependency "
                f"{node.name} belongs to peer chunk {owner.chunk_id} of "
                f"{body.owner.root_fqn!r}."
            )
        if owner.chunk_id == body.owner.chunk_id:
            dep_exchange_idx = exchange_indices.get(node)
            if dep_exchange_idx is not None and dep_exchange_idx > exchange_index:
                raise ValueError(
                    "ep_overlap token-exchange order is not topologically valid: "
                    f"launch {launch.name} for {body.owner} needs later "
                    f"same-chunk launch {node.name}."
                )

        visiting.add(node)
        for dep in sorted(
            _body_deps(node, body=body, owner_by_node=owner_by_node),
            key=order.__getitem__,
        ):
            dep_owner = _same_region_owner(
                dep,
                owner_by_node=owner_by_node,
                root_fqn=body.owner.root_fqn,
                is_backward=body.owner.is_backward,
            )
            visit(
                dep,
                allow_peer_chunk=allow_peer_chunk
                or (
                    dep_owner is not None and dep_owner.chunk_id != body.owner.chunk_id
                ),
            )
        visiting.remove(node)
        visited.add(node)
        closure.append(node)

    visit(launch)
    return tuple(sorted(closure, key=order.__getitem__))


def _ready_nodes(
    *,
    candidates_by_chunk: dict[int, set[fx.Node]],
    emitted: set[fx.Node],
    region: ChunkedRegion,
    chunk_order: tuple[int, ...],
    order: dict[fx.Node, int],
    owner_by_node: dict[fx.Node, ChunkOwner],
    include_waits: bool,
) -> tuple[fx.Node, ...]:
    """Return currently schedulable body nodes from candidate filler sets."""
    ready: list[fx.Node] = []
    for chunk_id in chunk_order:
        body = region.bodies_by_chunk[chunk_id]
        candidates = sorted(
            candidates_by_chunk.get(chunk_id, set()) - emitted,
            key=order.__getitem__,
        )
        for node in candidates:
            if not include_waits and _is_c10d_functional_node(node):
                continue
            deps = _body_deps(node, body=body, owner_by_node=owner_by_node)
            if all(dep in emitted for dep in deps):
                ready.append(node)
    return tuple(ready)


def _append_ready_blocks(
    blocks: list[tuple[fx.Node, ...]],
    emitted: set[fx.Node],
    *,
    candidates_by_chunk: dict[int, set[fx.Node]],
    region: ChunkedRegion,
    chunk_order: tuple[int, ...],
    order: dict[fx.Node, int],
    owner_by_node: dict[fx.Node, ChunkOwner],
    include_waits: bool,
) -> bool:
    """Append ready filler/tail blocks until the candidate frontier stalls."""
    made_progress = False
    while True:
        ready = tuple(
            node
            for node in _ready_nodes(
                candidates_by_chunk=candidates_by_chunk,
                emitted=emitted,
                region=region,
                chunk_order=chunk_order,
                order=order,
                owner_by_node=owner_by_node,
                include_waits=include_waits,
            )
            if node not in emitted
        )
        if not ready:
            return made_progress
        blocks.append(ready)
        emitted.update(ready)
        made_progress = True


def _build_region_phases(
    *,
    region: ChunkedRegion,
    exchanges_by_chunk: dict[int, tuple[_TokenExchange, ...]],
    order: dict[fx.Node, int],
    owner_by_node: dict[fx.Node, ChunkOwner],
) -> tuple[tuple[fx.Node, ...], ...]:
    """Step 4: construct wait-gated phases for one scheduled region."""
    chunk_order = (1, 0) if region.is_backward else (0, 1)
    exchange_indices = {
        chunk_id: {
            exchange.launch: idx
            for idx, exchange in enumerate(exchanges_by_chunk[chunk_id])
        }
        for chunk_id in chunk_order
    }
    closures = {
        chunk_id: tuple(
            _marker_closure(
                exchange.launch,
                body=region.bodies_by_chunk[chunk_id],
                order=order,
                owner_by_node=owner_by_node,
                exchange_index=idx,
                exchange_indices=exchange_indices[chunk_id],
            )
            for idx, exchange in enumerate(exchanges_by_chunk[chunk_id])
        )
        for chunk_id in chunk_order
    }
    closure_nodes = {
        chunk_id: {node for closure in chunk_closures for node in closure}
        for chunk_id, chunk_closures in closures.items()
    }
    filler = {
        chunk_id: set(region.bodies_by_chunk[chunk_id].nodes) - closure_nodes[chunk_id]
        for chunk_id in chunk_order
    }
    launch_nodes = {
        exchange.launch
        for chunk_exchanges in exchanges_by_chunk.values()
        for exchange in chunk_exchanges
    }

    def future_candidates(exchange_idx: int) -> dict[int, set[fx.Node]]:
        return {
            chunk_id: {
                node
                for closure in closures[chunk_id][exchange_idx + 1 :]
                for node in closure
                if node not in launch_nodes
            }
            | filler[chunk_id]
            for chunk_id in chunk_order
        }

    blocks: list[tuple[fx.Node, ...]] = []
    emitted: set[fx.Node] = set()
    for exchange_idx in range(len(exchanges_by_chunk[0])):
        for chunk_id in chunk_order:
            block = tuple(
                node for node in closures[chunk_id][exchange_idx] if node not in emitted
            )
            if block:
                blocks.append(block)
                emitted.update(block)
        _append_ready_blocks(
            blocks,
            emitted,
            candidates_by_chunk=future_candidates(exchange_idx),
            region=region,
            chunk_order=chunk_order,
            order=order,
            owner_by_node=owner_by_node,
            include_waits=False,
        )

    remaining = {
        chunk_id: set(region.bodies_by_chunk[chunk_id].nodes) - emitted
        for chunk_id in chunk_order
    }
    made_progress = True
    while made_progress:
        made_progress = False
        for chunk_id in chunk_order:
            made_progress |= _append_ready_blocks(
                blocks,
                emitted,
                candidates_by_chunk={chunk_id: remaining[chunk_id]},
                region=region,
                chunk_order=(chunk_id,),
                order=order,
                owner_by_node=owner_by_node,
                include_waits=True,
            )

    missing = [
        node
        for chunk_id in chunk_order
        for node in region.bodies_by_chunk[chunk_id].nodes
        if node not in emitted
    ]
    if missing:
        direction = "backward" if region.is_backward else "forward"
        raise ValueError(
            f"ep_overlap could not schedule all body nodes for {region.root_fqn!r} "
            f"({direction}); remaining: {', '.join(n.name for n in missing[:8])}."
        )
    logger.debug(
        "ep_overlap phases: root=%s direction=%s chunk_order=%s markers=%d "
        "phase_sizes=%s",
        region.root_fqn,
        "backward" if region.is_backward else "forward",
        chunk_order,
        len(exchanges_by_chunk[0]),
        [len(block) for block in blocks],
    )
    return tuple(blocks)


def _plan_region(
    region: ChunkedRegion,
    *,
    order: dict[fx.Node, int],
    owner_by_node: dict[fx.Node, ChunkOwner],
) -> _ScheduledRegion | None:
    """Steps 2-4: validate one chunked region and build its schedule phases."""
    root = region.root_fqn
    direction = "backward" if region.is_backward else "forward"
    if set(region.bodies_by_chunk) != {0, 1}:
        raise ValueError(
            f"ep_overlap expected both chunks for {root!r} ({direction}), "
            f"found {sorted(region.bodies_by_chunk)}."
        )

    exchanges_by_chunk = {
        chunk_id: _collect_token_exchanges(region.bodies_by_chunk[chunk_id])
        for chunk_id in (0, 1)
    }
    if not exchanges_by_chunk[0] and not exchanges_by_chunk[1]:
        logger.debug(
            "ep_overlap skipped region without token exchanges: root=%s direction=%s",
            root,
            direction,
        )
        return None
    if not exchanges_by_chunk[0] or not exchanges_by_chunk[1]:
        raise ValueError(
            f"ep_overlap found EP token exchanges for only one chunk of "
            f"{root!r} ({direction})."
        )
    if _exchange_signature(exchanges_by_chunk[0]) != _exchange_signature(
        exchanges_by_chunk[1]
    ):
        raise ValueError(
            f"ep_overlap expected matching EP all-to-all counts for "
            f"{root!r} ({direction}), found "
            f"chunk0={_exchange_signature(exchanges_by_chunk[0])} "
            f"chunk1={_exchange_signature(exchanges_by_chunk[1])}."
        )
    logger.debug(
        "ep_overlap planned region: root=%s direction=%s body_sizes=(%d,%d) "
        "marker_count=%d marker_labels=(%s,%s)",
        root,
        direction,
        len(region.bodies_by_chunk[0].nodes),
        len(region.bodies_by_chunk[1].nodes),
        len(exchanges_by_chunk[0]),
        _exchange_labels(exchanges_by_chunk[0]),
        _exchange_labels(exchanges_by_chunk[1]),
    )

    phases = _build_region_phases(
        region=region,
        exchanges_by_chunk=exchanges_by_chunk,
        order=order,
        owner_by_node=owner_by_node,
    )
    return _ScheduledRegion(region=region, phases=phases) if phases else None


def _move_nodes_in_order(graph: fx.Graph, nodes: list[fx.Node]) -> None:
    """Move graph nodes to an already topologically valid order."""
    cursor = None
    for node in nodes:
        if cursor is not None and cursor.next is not node:
            cursor.append(node)
        cursor = node


def _scheduled_node_order(
    graph: fx.Graph,
    scheduled_regions: list[_ScheduledRegion],
    order: dict[fx.Node, int],
) -> list[fx.Node]:
    """Step 5: combine scheduled region phases with ordinary graph topology."""
    owner: dict[fx.Node, tuple[_ScheduledRegion, int]] = {}
    anchors: dict[fx.Node, _ScheduledRegion] = {}
    for region in scheduled_regions:
        body_nodes = [node for phase in region.phases for node in phase]
        anchors[min(body_nodes, key=order.__getitem__)] = region
        for phase_idx, phase in enumerate(region.phases):
            for node in phase:
                owner[node] = (region, phase_idx)

    emitted: set[fx.Node] = set()
    new_order: list[fx.Node] = []
    completed_regions: set[int] = set()
    active_regions: set[int] = set()
    emit_stack: list[fx.Node] = []

    def emit_region(region: _ScheduledRegion) -> None:
        region_id = id(region)
        if region_id in completed_regions:
            return
        if region_id in active_regions:
            direction = "backward" if region.region.is_backward else "forward"
            raise ValueError(
                "ep_overlap found a cyclic dependency between scheduled regions "
                f"while emitting {region.region.root_fqn!r} ({direction})."
            )
        active_regions.add(region_id)
        for phase_idx, phase in enumerate(region.phases):
            for node in phase:
                emit(node, active_region=region, active_phase=phase_idx)
        active_regions.remove(region_id)
        completed_regions.add(region_id)

    def emit(
        node: fx.Node,
        *,
        active_region: _ScheduledRegion | None = None,
        active_phase: int | None = None,
    ) -> None:
        if node in emitted:
            return
        node_owner = owner.get(node)
        if node_owner is not None:
            if active_region is None or node_owner[0] is not active_region:
                emit_region(node_owner[0])
                return
            if active_phase is not None and node_owner[1] > active_phase:
                direction = (
                    "backward" if active_region.region.is_backward else "forward"
                )
                path = " -> ".join(n.name for n in (*emit_stack, node))
                raise ValueError(
                    "ep_overlap cannot emit requested schedule because "
                    f"{node.name} from phase {node_owner[1]} of "
                    f"{active_region.region.root_fqn!r} ({direction}) is needed "
                    f"before phase {active_phase}. Dependency path: {path}."
                )

        emit_stack.append(node)
        try:
            for inp in sorted(node.all_input_nodes, key=order.__getitem__):
                emit(inp, active_region=active_region, active_phase=active_phase)
            emitted.add(node)
            new_order.append(node)
        finally:
            emit_stack.pop()

    for node in list(graph.nodes):
        if node in emitted:
            continue
        region = anchors.get(node)
        if region is not None:
            emit_region(region)
        elif node not in owner:
            emit(node)

    if len(new_order) != len(list(graph.nodes)):
        raise AssertionError("ep_overlap manual block schedule dropped graph nodes")
    return new_order


def _apply_schedule(
    gm: fx.GraphModule,
    scheduled_regions: list[_ScheduledRegion],
    order: dict[fx.Node, int],
) -> None:
    """Step 5: apply scheduled order, lint, recompile, and validate phases."""
    _move_nodes_in_order(
        gm.graph,
        _scheduled_node_order(gm.graph, scheduled_regions, order),
    )
    gm.graph.lint()
    gm.recompile()
    new_order = ordered_nodes(gm)
    for region in scheduled_regions:
        previous_max: int | None = None
        for phase in region.phases:
            if not phase:
                continue
            phase_min = min(new_order[node] for node in phase)
            phase_max = max(new_order[node] for node in phase)
            if previous_max is not None and previous_max >= phase_min:
                direction = "backward" if region.region.is_backward else "forward"
                raise ValueError(
                    "ep_overlap failed to materialize requested block order for "
                    f"{region.region.root_fqn!r} ({direction})."
                )
            previous_max = phase_max


def _schedule_ep_overlap_regions(
    gm: fx.GraphModule,
    *,
    module_pattern: str,
    require_all_to_all: bool,
    reorder: bool = True,
) -> int:
    """Run validation or scheduling for all chunked regions matching a pattern."""
    order = ordered_nodes(gm)
    chunked_regions = collect_chunked_regions(gm, module_pattern=module_pattern)
    owner_by_node = {
        node: body.owner
        for region in chunked_regions
        for body in region.bodies_by_chunk.values()
        for node in body.nodes
    }
    scheduled_regions = [
        planned
        for region in chunked_regions
        if (
            planned := _plan_region(
                region,
                order=order,
                owner_by_node=owner_by_node,
            )
        )
        is not None
    ]
    logger.debug(
        "ep_overlap discovered %d chunked region(s), scheduled %d: pattern=%s",
        len(chunked_regions),
        len(scheduled_regions),
        module_pattern,
    )

    if scheduled_regions and reorder:
        _apply_schedule(gm, scheduled_regions, order)
    elif require_all_to_all:
        raise ValueError(
            f"ep_overlap did not find any chunked EP all-to-all regions for "
            f"pattern {module_pattern}."
        )
    return len(scheduled_regions)


def ep_overlap_validate_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    module_pattern: str,
    require_all_to_all: bool = False,
) -> fx.GraphModule:
    """Validate the already chunked graph without changing node order."""
    del example_inputs
    validated = _schedule_ep_overlap_regions(
        gm,
        module_pattern=module_pattern,
        require_all_to_all=require_all_to_all,
        reorder=False,
    )
    logger.info(
        "Validated %d ep_overlap chunked region(s): module=%s",
        validated,
        module_pattern,
    )
    return gm


def ep_overlap_schedule_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    module_pattern: str,
    require_all_to_all: bool = True,
) -> fx.GraphModule:
    """Reorder already chunked regions around EP all-to-alls."""
    del example_inputs
    scheduled = _schedule_ep_overlap_regions(
        gm,
        module_pattern=module_pattern,
        require_all_to_all=require_all_to_all,
    )
    logger.info(
        "Applied ep_overlap scheduling to %d chunked region(s): module=%s",
        scheduled,
        module_pattern,
    )
    return gm
