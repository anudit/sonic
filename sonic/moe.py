"""MoE routing economics.

Three questions the S1 plan turns on:
  1. How many distinct experts does a batch of b tokens touch?  -> bandwidth
  2. What does speculative decoding actually buy?                -> SKU table
  3. Does routing imbalance starve the systolic array?           -> P1 gate

(1) and (2) use an independent-uniform-routing model, which is a *pessimistic*
bound: real sequences show expert locality, so measured overlap should beat it.
Replacing this model with measured traces is the P0 deliverable -- see
p0/routing_trace.py.
"""

from __future__ import annotations

import numpy as np


def distinct_experts(n_experts: int, top_k: int, batch: int) -> float:
    """Expected distinct experts touched by `batch` independently-routed tokens.

    P(a given expert is missed by one token) = 1 - k/E, so the expected union is
    E * (1 - (1 - k/E)^batch).
    """
    return n_experts * (1.0 - (1.0 - top_k / n_experts) ** batch)


def expected_accepted(k: int, p: float) -> float:
    """Tokens accepted from a k-deep speculative chain at acceptance rate p.

    A chain of k drafts plus the verified base token yields
    sum_{i=0..k} p^i = (1 - p^(k+1)) / (1 - p).

    This is the term most often dropped: k+1 verify slots do NOT yield k+1
    tokens. Ignoring it overstates speculative gain by ~2x on this workload.
    """
    if p >= 1.0:
        return k + 1.0
    return (1.0 - p ** (k + 1)) / (1.0 - p)


def spec_decode_gain(model, k: int, p: float = 0.80,
                     drafter_mb: float = 53.0) -> dict:
    """Bytes per *accepted* token under speculative decoding.

    For a dense model each verify pass reads the weights once regardless of
    batch. For MoE the expert term grows with the union of routed experts, which
    is what erodes the gain.
    """
    solo = model.bytes_per_token()
    batch = k + 1
    acc = expected_accepted(k, p)

    if model.is_moe:
        exp_tot = model.expert_total_mb()
        non_expert = solo - exp_tot * model.top_k / model.n_experts
        frac = distinct_experts(model.n_experts, model.top_k, batch) / model.n_experts
        verify = exp_tot * frac + non_expert
        experts = distinct_experts(model.n_experts, model.top_k, batch)
    else:
        verify = solo
        experts = float("nan")

    per_token = (verify + k * drafter_mb) / acc
    return dict(k=k, batch=batch, experts=experts, verify_mb=verify,
                drafter_mb=k * drafter_mb, accepted=acc,
                mb_per_token=per_token, gain=solo / per_token)


def batch_gain(model, batch: int) -> dict:
    """True concurrent batching -- every token is real, none are rejected.

    This is where MoE actually pays: the expert union saturates at E but the
    batch keeps growing, so bytes/token falls roughly as 1/batch.
    """
    solo = model.bytes_per_token()
    if not model.is_moe:
        return dict(batch=batch, mb_per_token=solo / batch, gain=float(batch))
    exp_tot = model.expert_total_mb()
    non_expert = solo - exp_tot * model.top_k / model.n_experts
    frac = distinct_experts(model.n_experts, model.top_k, batch) / model.n_experts
    per_token = (exp_tot * frac + non_expert) / batch
    return dict(batch=batch, experts=distinct_experts(model.n_experts, model.top_k, batch),
                mb_per_token=per_token, gain=solo / per_token)


def route_counts(model, chunk: int, imbalance: float = 0.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Tokens landing on each expert for one prefill chunk of one layer.

    `imbalance` is the coefficient of variation of the expert-selection prior.
    0.0 is uniform routing; real MoE models typically sit at 0.3-0.8 even with
    load-balancing losses, which is exactly what the P1 occupancy gate probes.
    """
    rng = rng or np.random.default_rng(0)
    E, K = model.n_experts, model.top_k
    if imbalance <= 0:
        prior = np.full(E, 1.0 / E)
    else:
        # lognormal prior with the requested CV, renormalised to a distribution
        sigma = np.sqrt(np.log1p(imbalance ** 2))
        prior = rng.lognormal(-sigma ** 2 / 2, sigma, E)
        prior /= prior.sum()

    counts = np.zeros(E, dtype=np.int64)
    for _ in range(chunk):
        counts[rng.choice(E, size=K, replace=False, p=prior)] += 1
    return counts


def occupancy(counts: np.ndarray, tile: int) -> float:
    """Systolic-array occupancy for a set of ragged per-expert GEMMs.

    A tile x tile systolic array processes tokens in groups of `tile`. An expert
    holding n tokens occupies ceil(n/tile) passes but only fills n of the
    tile*ceil(n/tile) slots, so short experts waste the array.

    This is the P1 gate: >= 0.80 under measured routing imbalance.
    """
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    slots = np.ceil(counts / tile) * tile
    return float(counts.sum() / slots.sum())


def dspark_gain(model, drafter_params: float = 327.7e6, drafter_bits: float = 8.0,
                block: int = 9, p: float = 0.80, verify_all: bool = True) -> dict:
    """LFM2.5-8B-A1B-DSpark: a block drafter, not a sequential chain.

    config.json says block_size=9 and carries a dflash_config naming five
    target_layer_ids, so the drafter emits a whole block of 9 candidates in ONE
    pass while consuming the target's intermediate hidden states. It also has no
    vocab table of its own -- 327.7 M parameters against a 274.5 M body means the
    128K x 2048 embedding is shared with the target.

    Both facts matter enormously and both break the naive chain model:
      * drafter cost is 1 pass per cycle, not k passes;
      * that pass is 328 M weights, not 328 M + a 262 M vocab table.

    Set verify_all=False to model a linear-chain acceptance instead of a block.
    """
    solo = model.bytes_per_token()
    batch = block + 1
    acc = expected_accepted(block, p) if not verify_all else (
        (1.0 - p ** (block + 1)) / (1.0 - p))

    if model.is_moe:
        exp_tot = model.expert_total_mb()
        non_expert = solo - exp_tot * model.top_k / model.n_experts
        frac = distinct_experts(model.n_experts, model.top_k, batch) / model.n_experts
        verify = exp_tot * frac + non_expert
        experts = distinct_experts(model.n_experts, model.top_k, batch)
    else:
        verify = solo
        experts = float("nan")

    draft = drafter_params * drafter_bits / 8 / 1e6
    per_token = (verify + draft) / acc
    return dict(block=block, batch=batch, experts=experts, verify_mb=verify,
                drafter_mb=draft, accepted=acc, mb_per_token=per_token,
                gain=solo / per_token, p=p)


def acceptance_for_gain(model, target_gain: float, **kw) -> float:
    """What per-token acceptance rate does a claimed speedup imply?"""
    lo, hi = 0.01, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2
        if dspark_gain(model, p=mid, **kw)["gain"] < target_gain:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def measured_distinct(trace, batch: int, stride: int | None = None) -> float:
    """Mean distinct experts over sliding windows of a real routing trace.

    Use this instead of distinct_experts() wherever a measurement exists. On
    LFM2.5-8B-A1B the two differ by 1.2-1.6x -- real sequences reuse experts,
    the uniform model assumes they do not, and the gap is worth ~40% of the
    speculative-decode gain.
    """
    import numpy as np
    stride = stride or max(1, batch // 2)
    windows = range(0, len(trace) - batch + 1, stride)
    return float(np.mean([len(np.unique(trace[i:i + batch])) for i in windows]))


def dspark_gain_measured(model, trace, drafter_params: float = 327.7e6,
                         drafter_bits: float = 8.0, block: int = 9,
                         p: float = 0.80) -> dict:
    """DSpark economics against a measured trace rather than the uniform bound."""
    solo = model.bytes_per_token()
    batch = block + 1
    acc = expected_accepted(block, p)
    exp_tot = model.expert_total_mb()
    non_expert = solo - exp_tot * model.top_k / model.n_experts
    experts = measured_distinct(trace, batch)
    verify = exp_tot * experts / model.n_experts + non_expert
    draft = drafter_params * drafter_bits / 8 / 1e6
    per_token = (verify + draft) / acc
    return dict(block=block, batch=batch, experts=experts, verify_mb=verify,
                drafter_mb=draft, accepted=acc, mb_per_token=per_token,
                gain=solo / per_token, p=p, source="measured")
