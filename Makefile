.PHONY: demo-data p3-top p3-layer p3-generate p2-conv all test golden p0 p0-gates p0-gates-uniform p0-accbound p1 p1-dram p2 p2-sweep p2-units p2-router p2-pwl-sweep p3 p4-router p4-tile p4-sram p4-seq p4-top p4-top-hier p4-router-ci p4-pull vectors iv wave numbers clean

# Homebrew's binutils shadows Apple's ar with GNU ar, whose archives macOS ld
# rejects. Verilator links fail without this.
AR_FIX := AR=/usr/bin/ar
VROOT  := $(shell verilator --getenv VERILATOR_ROOT 2>/dev/null)

all: test numbers

test: golden
	@.venv/bin/python p2/lint_defs.py
	@cd tests && python3 run.py

golden:
	@cc -O2 -Wall -Wextra -o build/golden p0/golden/sonic_golden.c p0/golden/test_golden.c -lm 2>/dev/null \
	  || (mkdir -p build && cc -O2 -Wall -Wextra -o build/golden p0/golden/sonic_golden.c p0/golden/test_golden.c -lm)
	@./build/golden

# --- P0: numerics and routing freeze ---
p0:
	@python3 p0/recipe.py
	@python3 p0/routing_trace.py synthetic
	@python3 p0/dspark.py

# P0-1 quality gates: the recipe vs BF16 on the real checkpoint.
# CORPUS defaults to WikiText-2 via `datasets`; point it at a text file instead.
CORPUS ?=
p0-gates:
	@.venv/bin/python p0/gates.py $(if $(CORPUS),--corpus $(CORPUS),) --max-tokens $(or $(TOKENS),65536)

# Ablation: is the recipe's promotion of attention/routers/embedding earning
# its 0.39 extra bits, or would flat INT4 clear the gates just as well?
p0-gates-uniform:
	@.venv/bin/python p0/gates.py $(if $(CORPUS),--corpus $(CORPUS),) --uniform

# P0-5: local accumulator bounds against real activations.
p0-accbound:
	@.venv/bin/python p0/accbound.py --tokens $(or $(TOKENS),2048)

# --- P1: architecture model ---
p1:
	@python3 p1/occupancy.py
	@python3 p1/sweep.py --real-trace p0/out/real_routing.npz

# --- P2: unit RTL, differential benches, PPA loop ---
p2: build/sonic_golden.o
	@verilator --lint-only -Wall -Ip2/rtl p2/rtl/sonic_pe.sv --top-module sonic_pe
	@verilator --cc --exe -O2 -Wall -Wno-DECLFILENAME -Ip2/rtl \
	  -CFLAGS "-I$(CURDIR)/p0/golden -O2" --Mdir build/obj_acc \
	  --top-module sonic_acc p2/rtl/sonic_acc.sv p2/tb/tb_acc.cpp \
	  $(CURDIR)/build/sonic_golden.o >/dev/null
	@$(MAKE) -C build/obj_acc -f Vsonic_acc.mk $(AR_FIX) -j8 >/dev/null
	@./build/obj_acc/Vsonic_acc
	@$(MAKE) --no-print-directory p2-units
	@$(MAKE) --no-print-directory iv
	@python3 p2/ppa/loop.py

# P2-5 units. sonic_pe and sonic_softmax link the C golden model; tile and
# lmhead are self-checking.
p2-units: build/sonic_golden.o
	@for u in pe softmax; do \
	  rm -rf build/obj_$$u; \
	  verilator --cc --exe -O2 -Wall -Wno-DECLFILENAME -Ip2/rtl \
	    -CFLAGS "-I$(CURDIR)/p0/golden -O2" --Mdir build/obj_$$u \
	    --top-module sonic_$$u p2/rtl/sonic_acc.sv p2/rtl/sonic_$$u.sv \
	    p2/tb/tb_$$u.cpp $(CURDIR)/build/sonic_golden.o >/dev/null 2>&1; \
	  $(MAKE) -C build/obj_$$u -f Vsonic_$$u.mk $(AR_FIX) -j8 >/dev/null 2>&1; \
	  printf "  sonic_%-8s " "$$u"; ./build/obj_$$u/Vsonic_$$u | tail -1; \
	done
	@rm -rf build/obj_streamer
	@verilator --cc --exe -O2 -Wno-fatal -Ip2/rtl -CFLAGS "-O2" \
	  --Mdir build/obj_streamer --top-module sonic_streamer -GLANES=16 -GGROUP=64 \
	  p2/rtl/sonic_streamer.sv p2/tb/tb_streamer.cpp >/dev/null 2>&1
	@$(MAKE) -C build/obj_streamer -f Vsonic_streamer.mk $(AR_FIX) -j8 >/dev/null 2>&1
	@printf "  sonic_streamer "; ./build/obj_streamer/Vsonic_streamer | tail -1
	@rm -rf build/obj_seq
	@verilator --cc --exe -O2 -Wno-fatal -Ip2/rtl -CFLAGS "-O2" \
	  --Mdir build/obj_seq --top-module sonic_seq \
	  p2/rtl/sonic_seq.sv p2/tb/tb_seq.cpp >/dev/null 2>&1
	@$(MAKE) -C build/obj_seq -f Vsonic_seq.mk $(AR_FIX) -j8 >/dev/null 2>&1
	@printf "  sonic_seq      "; ./build/obj_seq/Vsonic_seq | tail -1
	@for u in tile lmhead rv32; do \
	  rm -rf build/obj_$$u; \
	  verilator --cc --exe -O2 -Wall -Wno-DECLFILENAME -Ip2/rtl \
	    --Mdir build/obj_$$u --top-module sonic_$$u \
	    p2/rtl/sonic_$$u.sv p2/tb/tb_$$u.cpp >/dev/null 2>&1; \
	  $(MAKE) -C build/obj_$$u -f Vsonic_$$u.mk $(AR_FIX) -j8 >/dev/null 2>&1; \
	  printf "  sonic_%-8s " "$$u"; ./build/obj_$$u/Vsonic_$$u | tail -1; \
	done
	@rm -rf build/obj_vec
	@verilator --cc --exe -O2 -Wno-fatal -Ip2/rtl \
	  --Mdir build/obj_vec --top-module sonic_vec \
	  p2/rtl/sonic_vec.sv p2/tb/tb_vec.cpp >/dev/null 2>&1
	@$(MAKE) -C build/obj_vec -f Vsonic_vec.mk $(AR_FIX) -j8 >/dev/null 2>&1
	@printf "  sonic_%-8s " "vec"; ./build/obj_vec/Vsonic_vec | tail -1
	@rm -rf build/obj_sram_gate
	@verilator --cc --exe -O2 -Wall -Wno-DECLFILENAME -Ip2/rtl -CFLAGS "-O2" \
	  --Mdir build/obj_sram_gate --top-module sonic_sram_gate \
	  p2/rtl/sonic_sram_gate.sv p2/tb/tb_sram_gate.cpp >/dev/null 2>&1
	@$(MAKE) -C build/obj_sram_gate -f Vsonic_sram_gate.mk $(AR_FIX) -j8 >/dev/null 2>&1
	@printf "  sonic_%-8s " "sram_gate"; ./build/obj_sram_gate/Vsonic_sram_gate | tail -1

# Router RTL driven by REAL tensors from the 8.47B checkpoint.
p2-router: p2/vectors/router_l5.bin
	@rm -rf build/obj_router
	@verilator --cc --exe -O2 -Wall -Wno-DECLFILENAME -Ip2/rtl \
	  -CFLAGS "-DSEGS_OVERRIDE=32 -DRANGE_OVERRIDE=4 -O2" --Mdir build/obj_router \
	  --top-module sonic_router p2/rtl/sonic_router.sv p2/tb/tb_router.cpp >/dev/null
	@$(MAKE) -C build/obj_router -f Vsonic_router.mk $(AR_FIX) -j8 >/dev/null
	@./build/obj_router/Vsonic_router p2/vectors/router_l5.bin 512

# Size the sigmoid PWL against measured routing agreement, over BOTH axes.
#
# Finding 20 (the FFN's SiLU) showed the input RANGE is the lever and the
# segment COUNT is not. This sweeps the router's sigmoid the same way: the
# router's logits sit inside about +-4, so a table spanning [-8, 8) spends most
# of its segments on inputs that never occur.
p2-pwl-sweep: p2/vectors/router_l5.bin
	@echo "  range   segs   top-1    top-4 set   exact order"
	@for R in 8 4 2; do \
	for S in 8 16 32 64; do \
	  rm -rf build/obj_r$$R_$$S; \
	  verilator --cc --exe -O2 -Wno-fatal -Wno-DECLFILENAME -Ip2/rtl \
	    -DROUTER_PWL_SEGS=$$S -DROUTER_PWL_RANGE=$$R \
	    -CFLAGS "-DSEGS_OVERRIDE=$$S -DRANGE_OVERRIDE=$$R -O2" \
	    --Mdir build/obj_r$$R_$$S --top-module sonic_router \
	    p2/rtl/sonic_router.sv p2/tb/tb_router.cpp >/dev/null 2>&1; \
	  $(MAKE) -C build/obj_r$$R_$$S -f Vsonic_router.mk $(AR_FIX) -j8 >/dev/null 2>&1; \
	  O=$$(./build/obj_r$$R_$$S/Vsonic_router p2/vectors/router_l5.bin 512 2>/dev/null); \
	  printf "  +-%-4s  %4s   %s   %s      %s\n" "$$R" "$$S" \
	    "$$(echo "$$O" | grep 'top-1' | awk '{print $$4}')" \
	    "$$(echo "$$O" | grep 'set match' | awk '{print $$4}')" \
	    "$$(echo "$$O" | grep 'exact order' | awk '{print $$4}')"; \
	done; done

# Second simulator, independent front end, plus a VCD for GTKWave.
iv:
	@mkdir -p build
	@iverilog -g2012 -I p2/rtl -o build/tb_acc_iv p2/tb/tb_acc_iv.sv p2/rtl/sonic_acc.sv
	@./build/tb_acc_iv

wave: iv
	@gtkwave build/acc.vcd 2>/dev/null &

p2/vectors/router_l5.bin:
	@.venv/bin/python p3/export_vectors.py --layer 5 --tokens 512

vectors: p2/vectors/router_l5.bin

p2-sweep:
	@python3 p2/ppa/loop.py --unit sonic_acc --sweep ACC_LOCAL=12,16,20,24
	@python3 p2/ppa/loop.py --unit sonic_acc --sweep ACC_FOLD=4,8,16,32

build/sonic_golden.o: p0/golden/sonic_golden.c p0/golden/sonic_golden.h
	@mkdir -p build && cc -O2 -c -o $@ $<

# --- P3: integration, real-model capture ---
p3:
	@.venv/bin/python p3/capture_routing.py --tokens 4096
	@python3 p0/routing_trace.py trace --trace p0/out/real_routing.npz

# --- P4: block-level place and route (needs Nix + OpenLane 2, see p4/README.md)
p4-router:
	@command -v nix >/dev/null || { echo "Nix not installed -- see p4/README.md"; exit 1; }
	@cd p4/openlane/router && nix run github:librelane/librelane -- config.json
	@echo "opening the routed layout in KLayout"
	@open -a KLayout $$(ls -td p4/openlane/router/runs/*/final/gds/*.gds | head -1)

p4-tile:
	@command -v nix >/dev/null || { echo "Nix not installed -- see p4/README.md"; exit 1; }
	@cd p4/openlane/tile && nix run github:librelane/librelane -- config.json

p4-sram:
	@command -v nix >/dev/null || { echo "Nix not installed -- see p4/README.md"; exit 1; }
	@cd p4/openlane/sram && nix run github:librelane/librelane -- config.json

p4-seq:
	@command -v nix >/dev/null || { echo "Nix not installed -- see p4/README.md"; exit 1; }
	@cd p4/openlane/seq && nix run github:librelane/librelane -- config.json

# Full-chip flat smoke run: all 18 RTL files, every SYNTH_PARAMETERS knob at
# its smallest tractable value (see p4/openlane/top/config.json's own "//
# scope" comment). This is NOT the shipping-scale chip -- see p4/HANDOFF.md
# T4.2 -- it proves the full hierarchy synthesizes-to-GDS as one netlist.
# Multi-hour, x86-64 Linux + Nix only (see p4/README.md).
p4-top:
	@command -v nix >/dev/null || { echo "Nix not installed -- see p4/README.md"; exit 1; }
	@cd p4/openlane/top && nix run github:librelane/librelane -- config.json
	@echo "opening the routed layout in KLayout"
	@open -a KLayout $$(ls -td p4/openlane/top/runs/*/final/gds/*.gds | head -1)

# Hierarchical assembly (HANDOFF.md T4.2): sonic_tile/sonic_router/
# sonic_sram_bank consumed as pre-hardened macros via MACROS, not
# re-synthesized -- see p4/openlane/top-hier/HANDOFF_TOP_HIER.md for the
# prerequisite GDS/LEF/lib artifacts this needs before it will run at all.
p4-top-hier:
	@command -v nix >/dev/null || { echo "Nix not installed -- see p4/README.md"; exit 1; }
	@cd p4/openlane/top-hier && nix run github:librelane/librelane -- config.json
	@echo "opening the routed layout in KLayout"
	@open -a KLayout $$(ls -td p4/openlane/top-hier/runs/*/final/gds/*.gds | head -1)

# CI path: no Nix on this machine. Runs on an x86-64 Linux runner, which is
# where OpenROAD actually has support. LANES/PWL override the config defaults.
LANES ?= 16
PWL   ?= 64

RUN_LATEST = gh run list --workflow=router-gds.yml -L1 --json databaseId -q '.[0].databaseId'

# Dispatch, then wait for a run id that is not the one that was newest before
# dispatch -- GitHub takes a few seconds to register the run, and grabbing the
# newest id too early watches the PREVIOUS run to completion instead.
p4-router-ci:
	@PREV=$$($(RUN_LATEST)); \
	 gh workflow run router-gds.yml -f lanes=$(LANES) -f pwl_segs=$(PWL); \
	 for i in $$(seq 1 30); do \
	   ID=$$($(RUN_LATEST)); \
	   [ "$$ID" != "$$PREV" ] && break; \
	   sleep 2; \
	 done; \
	 echo "watching run $$ID"; gh run watch $$ID

# Pull the newest run that actually produced artifacts, not merely the newest.
p4-pull:
	@ID=$$(gh run list --workflow=router-gds.yml -L20 --status success \
		--json databaseId -q '.[0].databaseId'); \
	 [ -n "$$ID" ] || { echo "no successful run yet"; exit 1; }; \
	 rm -rf p4/out && mkdir -p p4/out; \
	 gh run download $$ID -D p4/out; \
	 echo "--- pulled from run $$ID ---"; \
	 find p4/out \( -name '*.gds' -o -name '*.png' \) | sed 's/^/  /'; \
	 PNG=$$(find p4/out -name '*.png' | head -1); \
	 [ -n "$$PNG" ] && open "$$PNG" || true

numbers:
	@python3 -m sonic.report

clean:
	@rm -rf build p0/out p1/out __pycache__ */__pycache__

# --- P1-3: DRAM efficiency of the expert-gather pattern (needs dramsim3 built)
DRAM_OUT   := p1/out/dram
DRAM_SIM   := dramsim3/build/dramsim3main
DRAM_CFG   := configs/LPDDR4_8Gb_x16_2400.ini
DRAM_LINES ?= 200000
DRAM_CYC   ?= 100000

$(DRAM_SIM):
	@cd dramsim3/build && cmake .. -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >/dev/null && make -j8 >/dev/null
	@echo "built $@"

p1-dram: $(DRAM_SIM)
	@mkdir -p $(DRAM_OUT)
	@for p in stream expert scatter; do \
	  python3 p1/dram_trace.py --pattern $$p --lines $(DRAM_LINES) \
	    --out $(DRAM_OUT)/$$p.trace >/dev/null; \
	  mkdir -p $(DRAM_OUT)/out_$$p; \
	  (cd dramsim3 && ./build/dramsim3main $(DRAM_CFG) \
	     -t $(CURDIR)/$(DRAM_OUT)/$$p.trace \
	     -o $(CURDIR)/$(DRAM_OUT)/out_$$p -c $(DRAM_CYC) >/dev/null 2>&1); \
	done
	@python3 p1/dram.py

# sonic_conv is built at CH=4 so its packed ports fit a word; p2-units keeps
# the units whose benches need no parameter override.
p2-conv:
	@rm -rf build/obj_conv
	@verilator --cc --exe -O2 -Wno-fatal -Ip2/rtl -CFLAGS "-O2" \
	  --Mdir build/obj_conv --top-module sonic_conv -GCH=4 \
	  p2/rtl/sonic_conv.sv p2/tb/tb_conv.cpp >/dev/null 2>&1
	@$(MAKE) -C build/obj_conv -f Vsonic_conv.mk $(AR_FIX) -j8 >/dev/null 2>&1
	@./build/obj_conv/Vsonic_conv

# --- P3-5: one MoE layer end to end, RTL against the real model
p2/vectors/layer_l5.bin:
	@.venv/bin/python p3/export_layer.py --layer 5 --tokens 1

p3-layer: p2/vectors/layer_l5.bin build/sonic_golden.o
	@rm -rf build/obj_layer
	@verilator --cc --exe -O2 -Wno-fatal -Ip2/rtl \
	  -CFLAGS "-I$(CURDIR)/p0/golden -O2" --Mdir build/obj_layer \
	  --top-module sonic_tile p2/rtl/sonic_acc.sv p2/rtl/sonic_pe.sv \
	  p2/rtl/sonic_tile.sv p2/tb/tb_layer.cpp $(CURDIR)/build/sonic_golden.o >/dev/null 2>&1
	@$(MAKE) -C build/obj_layer -f Vsonic_tile.mk $(AR_FIX) -j8 >/dev/null 2>&1
	@./build/obj_layer/Vsonic_tile p2/vectors/layer_l5.bin $(or $(ROWS),256)

p2/vectors/layer_l0_dense.bin:
	@.venv/bin/python p3/export_layer.py --layer 0 --out $@

p3-dense: p2/vectors/layer_l0_dense.bin build/sonic_golden.o
	@rm -rf build/obj_layer
	@verilator --cc --exe -O2 -Wno-fatal -Ip2/rtl \
	  -CFLAGS "-I$(CURDIR)/p0/golden -O2" --Mdir build/obj_layer \
	  --top-module sonic_tile p2/rtl/sonic_acc.sv p2/rtl/sonic_pe.sv \
	  p2/rtl/sonic_tile.sv p2/tb/tb_layer.cpp $(CURDIR)/build/sonic_golden.o >/dev/null 2>&1
	@$(MAKE) -C build/obj_layer -f Vsonic_tile.mk $(AR_FIX) -j8 >/dev/null 2>&1
	@./build/obj_layer/Vsonic_tile $< 1

p3-ring:
	@python3 p3/producer.py
	@$(MAKE) p2-units

p2/vectors/multi/manifest.json:
	@.venv/bin/python p3/export_multi_layer.py --layers 5 6 7 8 --tokens 1

p3-top: p2/vectors/layer_l5.bin p2/vectors/multi/manifest.json build/sonic_golden.o
	@rm -rf build/obj_top
	@verilator --cc --exe -O2 -Wno-fatal -Ip2/rtl \
	  -CFLAGS "-I$(CURDIR)/p0/golden -O2" --Mdir build/obj_top \
	  --top-module sonic_top p2/rtl/sonic_acc.sv p2/rtl/sonic_pe.sv \
	  p2/rtl/sonic_conv.sv p2/rtl/sonic_softmax.sv p2/rtl/sonic_lmhead.sv \
	  p2/rtl/sonic_streamer.sv p2/rtl/sonic_router.sv p2/rtl/sonic_tile.sv \
	  p2/rtl/sonic_seq.sv p2/rtl/sonic_vec.sv p2/rtl/sonic_sram_bank.sv \
	  p2/rtl/sonic_sram_gate.sv p2/rtl/sonic_sram.sv p2/rtl/sonic_rv32.sv p2/rtl/sonic_noc.sv \
	  p2/rtl/sonic_mbist.sv p2/rtl/sonic_phy_lpddr5x.sv p2/rtl/sonic_ioring.sv \
	  p2/rtl/sonic_top.sv p2/tb/tb_top.cpp $(CURDIR)/build/sonic_golden.o >/dev/null 2>&1
	@$(MAKE) -C build/obj_top -f Vsonic_top.mk $(AR_FIX) -j8 >/dev/null 2>&1
	@./build/obj_top/Vsonic_top p2/vectors/layer_l5.bin

# --- demo: regenerate the floorplan's data and re-inject it
demo-data:
	@python3 demo/build.py
	@echo "demo/floorplan.html updated"

# --- P3: text in, text out under the chip's weight formats
p3-generate:
	@.venv/bin/python p3/generate.py --tokens $(or $(TOKENS),48)
