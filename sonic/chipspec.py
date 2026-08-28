"""Sonic S1 hardware parameters.

Every constant here is a P1 deliverable: replace the estimate with a measured or
vendor-quoted number and re-run the sweeps. Sources are noted per field.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ChipSpec:
    name: str

    # --- compute ---
    mac_lanes: int = 16384          # 128 x 128 systolic, dual-mode
    tile: int = 128                 # systolic edge; model dims must be multiples
    clk_ghz: float = 1.0
    boost_ghz: float = 1.4          # prefill-only DVFS point
    accum_banks: int = 8            # speculative verify depth
    attn_engine_gain: float = 2.5   # flash-attention unit vs the general array

    # --- memory ---
    sram_mb: float = 8.0
    sram_banks: int = 16
    bus_bits: int = 64              # LPDDR5X PHY width
    dram_gbps: float = 68.3
    dram_eff: float = 1.00          # achieved fraction of peak; 0.85 is the P1 gate
    dram_gb: int = 8

    # --- energy, pJ/bit and pJ/op ---
    e_dram_bit: float = 5.0e-12     # LPDDR5X I/O + DRAM core
    e_sram_bit: float = 0.1e-12
    e_mac_op: float = 0.05e-12      # INT4 x INT8 at 14nm
    p_leak_w: float = 0.55          # leakage + clocks + NoC, array gated
    p_stream_w: float = 0.30        # streamer, dequant, AES
    p_vector_w: float = 0.20

    # --- process ---
    node: str = "14nm LPP"
    sram_mm2_per_mb: float = 0.80
    mac_mm2_per_lane: float = 350e-6
    util: float = 0.65              # routing / utilisation overhead

    @property
    def tops(self) -> float:
        """INT4 throughput, 2 ops per MAC."""
        return self.mac_lanes * 2 * self.clk_ghz / 1e3

    @property
    def eff_gbps(self) -> float:
        return self.dram_gbps * self.dram_eff

    def lanes_to_saturate(self, bits: float = 4.60) -> float:
        """Lanes that batch-1 GEMV can keep busy at this bandwidth.

        Batch-1 GEMV consumes exactly one MAC per weight, so this is the point
        beyond which extra lanes only serve prefill or a verify batch.
        """
        return self.eff_gbps * 1e9 * 8 / bits / (self.clk_ghz * 1e9)

    def sku(self, bus_bits: int, dram_gbps: float, dram_gb: int, name: str) -> "ChipSpec":
        return replace(self, name=name, bus_bits=bus_bits,
                       dram_gbps=dram_gbps, dram_gb=dram_gb)


S1 = ChipSpec(name="S1")

SKUS = {
    "A": S1.sku(32, 25.6, 8, "S1-A wearable"),
    "B": S1.sku(64, 68.3, 8, "S1-B handheld"),
    "C": S1.sku(128, 136.5, 8, "S1-C edge"),
}

# Area model, mm2 at 14nm LPP. Blocks whose size is fixed by IP or IO rather
# than by the sweeps are listed as constants; array and SRAM are computed.
AREA_FIXED = {
    "LPDDR5X x64 PHY": 2.09,
    "Programmable vector unit": 1.50,
    "Flash-attention engine": 0.80,
    "Streamer + expert gather + dequant + AES": 0.45,
    "Conv + vector units": 0.43,
    "IO ring / ESD": 0.40,
    "PLLs, LDOs, PVT": 0.33,
    "RV32 + NoC + DFT": 0.29,
    "Token-gather / ragged-GEMM scheduler": 0.25,
    "LM head + sampler": 0.14,
    "MoE router + top-k": 0.05,
}
PHY_AREA = {32: 1.05, 64: 2.09, 128: 4.18}
