#!/usr/bin/env python3
"""P0-3: what LFM2.5-8B-A1B-DSpark is actually worth on S1.

Liquid publishes "~2.5x faster decoding with identical outputs". That number is
real, and it does not transfer to this chip unchanged -- because it is measured
against a different bottleneck.

  On a GPU, batch-1 decode is largely latency-bound: many small sequential
  kernel launches with the weights already in HBM. Speculation replaces 9
  launches with 1, so the win is occupancy and launch overhead.

  On S1, batch-1 decode is purely DRAM-bandwidth-bound (see the roofline: 119
  lanes saturate the array of 16,384). Speculation only helps if it reduces
  BYTES. On a dense model it does, exactly. On an MoE it mostly does not,
  because a verify batch of 10 routes to ~24 of 32 experts.

This script sizes the real gain and, just as importantly, enumerates what the
drafter demands from the hardware. Run it before assuming a published speedup.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sonic import load  # noqa: E402
from sonic.moe import acceptance_for_gain, distinct_experts, dspark_gain  # noqa: E402

CFG = Path(__file__).resolve().parent.parent / "configs" / "lfm2.5-8b-a1b-dspark.json"
DRAFTER_PARAMS = 327.7e6      # from the safetensors blob size, bf16
DRAFTER_BODY = 274.5e6        # 5 layers + markov head, computed from config


def main() -> int:
    m = load("lfm2.5-8b-a1b")
    c = json.loads(CFG.read_text())
    block = c["block_size"]
    taps = c["dflash_config"]["target_layer_ids"]

    print(f"drafter: {c['num_hidden_layers']} dense attention layers, "
          f"d={c['hidden_size']}, ffn={c['intermediate_size']}, block={block}")
    print(f"  taps target layers {taps} of {c['dflash_config']['num_target_layers']}")
    print(f"  {DRAFTER_PARAMS/1e6:.1f} M params vs a {DRAFTER_BODY/1e6:.1f} M body")
    print(f"  -> the 128K x 2048 vocab table is SHARED with the target, not carried\n")

    print(f"solo decode: {m.bytes_per_token():.0f} MB/token")
    print(f"verify batch {block+1} touches "
          f"{distinct_experts(m.n_experts, m.top_k, block+1):.1f} of {m.n_experts} experts "
          f"({distinct_experts(m.n_experts, m.top_k, block+1)/m.n_experts:.0%})\n")

    print(f"  {'drafter':>9}{'p':>7}{'verify':>9}{'draft':>8}{'accepted':>10}{'MB/tok':>9}{'gain':>7}")
    for bits, lab in ((8.0, "INT8"), (4.25, "INT4")):
        for p in (0.70, 0.80, 0.90, 0.95):
            g = dspark_gain(m, DRAFTER_PARAMS, bits, block, p)
            print(f"  {lab:>9}{p:>7.2f}{g['verify_mb']:>9.0f}{g['drafter_mb']:>8.0f}"
                  f"{g['accepted']:>10.2f}{g['mb_per_token']:>9.0f}{g['gain']:>7.2f}x")

    print()
    for bits, lab in ((8.0, "INT8"), (4.25, "INT4")):
        p = acceptance_for_gain(m, 2.5, drafter_bits=bits, block=block)
        print(f"  reaching 2.5x with an {lab} drafter needs p = {p:.3f}"
              + ("   <- not plausible; the published figure is a different bottleneck"
                 if p > 0.97 else ""))

    print("\n--- what the drafter DEMANDS of the hardware ---")
    kv = c["num_hidden_layers"] * 2 * c["num_key_value_heads"] * c["head_dim"] / 1e3
    reqs = [
        (f"hidden-state tap at target layers {taps}",
         f"stage 5 x batch x {m.d} B in SRAM; {5*(block+1)*m.d/1e3:.0f} KB at block {block}"),
        ("second KV cache for the drafter's 5 dense attention layers",
         f"{kv:.1f} KB/token on top of the target's {m.kv_kb_per_token():.1f} KB -- "
         f"nearly doubles KV"),
        ("drafter weights resident in DRAM",
         f"{DRAFTER_PARAMS*8/8/1e6:.0f} MB at INT8 -- fits the 8 GB part"),
        ("mask token id 125017 in the sampler path", "special-token handling in the LM head unit"),
        ("confidence head -> adaptive block length",
         "firmware must vary the block, so the verify batch is dynamic, not fixed"),
        ("rope_is_neox_style=false, rope_theta=5e6",
         "a SECOND RoPE style; the CORDIC unit needs both, not just the target's"),
    ]
    for what, why in reqs:
        print(f"  * {what}\n      {why}")

    print("\nverdict: budget 1.1-1.7x on S1, not 2.5x. It is lossless, so it costs")
    print("nothing in quality -- but it is not the lever the SKU table should lean on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
