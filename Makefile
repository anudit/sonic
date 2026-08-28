.PHONY: all test golden p0 p1 p2 p2-sweep p2-units p2-router p2-pwl-sweep p3 p4-router p4-router-ci p4-pull vectors iv wave numbers clean

# Homebrew's binutils shadows Apple's ar with GNU ar, whose archives macOS ld
# rejects. Verilator links fail without this.
AR_FIX := AR=/usr/bin/ar
VROOT  := $(shell verilator --getenv VERILATOR_ROOT 2>/dev/null)

all: test numbers

test: golden
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

# --- P1: architecture model ---
p1:
	@python3 p1/occupancy.py
	@python3 p1/sweep.py --imbalance 0.5

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

# P2-5 units: dual-mode systolic tile and the streaming LM head.
p2-units:
	@for u in tile lmhead; do \
	  rm -rf build/obj_$$u; \
	  verilator --cc --exe -O2 -Wall -Wno-DECLFILENAME -Ip2/rtl \
	    --Mdir build/obj_$$u --top-module sonic_$$u \
	    p2/rtl/sonic_$$u.sv p2/tb/tb_$$u.cpp >/dev/null 2>&1; \
	  $(MAKE) -C build/obj_$$u -f Vsonic_$$u.mk $(AR_FIX) -j8 >/dev/null 2>&1; \
	  printf "  sonic_%-8s " "$$u"; ./build/obj_$$u/Vsonic_$$u | tail -1; \
	done

# Router RTL driven by REAL tensors from the 8.47B checkpoint.
p2-router: p2/vectors/router_l5.bin
	@rm -rf build/obj_router
	@verilator --cc --exe -O2 -Wall -Wno-DECLFILENAME -Ip2/rtl \
	  -CFLAGS "-DSEGS_OVERRIDE=64 -O2" --Mdir build/obj_router \
	  --top-module sonic_router p2/rtl/sonic_router.sv p2/tb/tb_router.cpp >/dev/null
	@$(MAKE) -C build/obj_router -f Vsonic_router.mk $(AR_FIX) -j8 >/dev/null
	@./build/obj_router/Vsonic_router p2/vectors/router_l5.bin 512

# Size the sigmoid PWL against measured routing agreement.
p2-pwl-sweep: p2/vectors/router_l5.bin
	@echo "  segs   top-1    top-4 set   exact order"
	@for S in 16 32 64 128; do \
	  rm -rf build/obj_r$$S; \
	  verilator --cc --exe -O2 -Wno-fatal -Wno-DECLFILENAME -Ip2/rtl \
	    -DROUTER_PWL_SEGS=$$S -CFLAGS "-DSEGS_OVERRIDE=$$S -O2" \
	    --Mdir build/obj_r$$S --top-module sonic_router \
	    p2/rtl/sonic_router.sv p2/tb/tb_router.cpp >/dev/null 2>&1; \
	  $(MAKE) -C build/obj_r$$S -f Vsonic_router.mk $(AR_FIX) -j8 >/dev/null 2>&1; \
	  O=$$(./build/obj_r$$S/Vsonic_router p2/vectors/router_l5.bin 512 2>/dev/null); \
	  printf "  %4s   %s   %s      %s\n" "$$S" \
	    "$$(echo "$$O" | grep 'top-1' | awk '{print $$4}')" \
	    "$$(echo "$$O" | grep 'set match' | awk '{print $$4}')" \
	    "$$(echo "$$O" | grep 'exact order' | awk '{print $$4}')"; \
	done

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

# CI path: no Nix on this machine. Runs on an x86-64 Linux runner, which is
# where OpenROAD actually has support. LANES/PWL override the config defaults.
LANES ?= 16
PWL   ?= 64

p4-router-ci:
	@gh workflow run router-gds.yml -f lanes=$(LANES) -f pwl_segs=$(PWL)
	@sleep 5
	@gh run watch $$(gh run list --workflow=router-gds.yml -L1 --json databaseId -q '.[0].databaseId')

p4-pull:
	@rm -rf p4/out && mkdir -p p4/out
	@gh run download $$(gh run list --workflow=router-gds.yml -L1 \
		--json databaseId -q '.[0].databaseId') -D p4/out
	@echo "--- pulled ---" && find p4/out -name '*.gds' -o -name '*.png' | sed 's/^/  /'
	@open p4/out/*/sonic_router.png 2>/dev/null || true

numbers:
	@python3 -m sonic.report

clean:
	@rm -rf build p0/out p1/out __pycache__ */__pycache__
