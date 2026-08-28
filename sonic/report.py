"""Print every headline number in the plan. `make numbers` or `python -m sonic.report`."""

from . import SKUS, decode, load, prefill
from .moe import batch_gain, dspark_gain, dspark_gain_measured, measured_distinct
from .quant import UNIFORM_INT4, fmt
from .roofline import area, min_chunk_for_array


def _trace():
    """Real routing decisions, if p3/capture_routing.py has been run."""
    import numpy as np
    from pathlib import Path
    f = Path(__file__).resolve().parent.parent / "p0" / "out" / "real_routing.npz"
    return np.load(f)["routing"].astype(int) if f.exists() else None


def main() -> None:
    m8, m2 = load("lfm2.5-8b-a1b"), load("lfm2.5-2.6b")

    print("=" * 74)
    print("SONIC S1 -- headline numbers".center(74))
    print("=" * 74)

    for m in (m8, m2):
        print(f"\n{m.name}  {m.n_layers} layers ({m.n_conv} conv / {m.n_attn} attn), "
              f"d={m.d}, vocab={m.vocab:,}")
        print(f"  {'block':30s}{'total':>12}{'active':>12}{'format':>9}")
        for b in m.blocks:
            print(f"  {b.name:30s}{b.total/1e6:>11.1f}M{b.active/1e6:>11.1f}M"
                  f"{fmt(b.name).kind:>9}")
        print(f"  {'':30s}{m.total_params/1e6:>11.1f}M{m.active_params/1e6:>11.1f}M"
              f"   ({m.sparsity:.1%} active)")
        print(f"  traffic {m.bytes_per_token():.0f} MB/token @ {m.avg_bits():.2f} bits"
              f"   (uniform-INT4 ideal: {m.bytes_per_token(UNIFORM_INT4):.0f} MB)")
        print(f"  resident {m.resident_mb():,.0f} MB | {m.gop_per_token():.2f} GOP/token"
              f" | KV {m.kv_kb_per_token():.2f} KB/token")

    print("\n" + "-" * 74)
    print(f"{'SKU':>4}{'GB/s':>8}{'8B-A1B':>10}{'2.6B':>9}{'power':>8}{'mJ/tok':>9}{'die mm2':>10}")
    for k, c in SKUS.items():
        d8, d2 = decode(m8, c), decode(m2, c)
        print(f"{k:>4}{c.dram_gbps:>8.1f}{d8.tok_s:>9.1f}t{d2.tok_s:>8.1f}t"
              f"{d8.watts:>7.2f}W{d8.mj_per_token:>9.0f}{area(c)['_total']:>10.2f}")

    b = SKUS["B"]
    print(f"\narray {b.tops:.1f} TOPS | batch-1 GEMV saturates at "
          f"{b.lanes_to_saturate():.0f} lanes -> {b.mac_lanes/b.lanes_to_saturate():.0f}x headroom "
          f"for prefill")
    print(f"min prefill chunk for a {b.tile}-edge tile: {min_chunk_for_array(m8, b)}")

    print(f"\nTTFT, SKU B:  {'prompt':>8}{'8B-A1B':>10}{'2.6B':>10}  bound")
    for P in (128, 512, 2048, 8192, 32768):
        a, d = prefill(m8, b, P), prefill(m2, b, P)
        print(f"{'':14}{P:>8,}{a.ttft_ms:>9.0f}ms{d.ttft_ms:>9.0f}ms  {a.bound}")

    tr = _trace()
    src = "MEASURED routing" if tr is not None else "uniform bound (no trace; run `make p3`)"
    print(f"\nspeculative decode (DSpark block=9, INT8 drafter) -- {src}:")
    for p in (0.70, 0.80, 0.90):
        g = dspark_gain_measured(m8, tr, p=p) if tr is not None else dspark_gain(m8, p=p)
        print(f"  p={p:.2f}: {g['mb_per_token']:>6.0f} MB/accepted-token  {g['gain']:.2f}x"
              + (f"   (bound would say {dspark_gain(m8, p=p)['gain']:.2f}x)"
                 if tr is not None else ""))

    print("concurrent batching (where MoE actually pays):")
    for n in (8, 32, 128):
        b = batch_gain(m8, n)
        line = f"  batch {n:>3}: {b['mb_per_token']:>6.1f} MB/token  {b['gain']:.1f}x"
        if tr is not None and n <= 32:
            e = measured_distinct(tr, n)
            exp_tot = m8.expert_total_mb()
            ne = m8.bytes_per_token() - exp_tot * m8.top_k / m8.n_experts
            pm = (exp_tot * e / m8.n_experts + ne) / n
            line += f"   measured {pm:.1f} MB  {m8.bytes_per_token()/pm:.1f}x"
        print(line)


if __name__ == "__main__":
    main()
