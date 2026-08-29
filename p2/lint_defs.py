#!/usr/bin/env python3
"""P2-13: level-2 parameterization as a lint check, not a convention.

Every `define in sonic_defs.svh that some flow overrides at synth/sweep time
(p2/ppa/loop.py's REQUIRED_DEFINES, or any p4/openlane/*/config.json's
VERILOG_DEFINES / SYNTH_PARAMETERS) MUST be guarded with `` `ifndef `` in
sonic_defs.svh, or the override silently does nothing (a second, unguarded
`` `define `` after the file's already been read wins or errors depending on
tool, and either way the sweep is measuring the default, not what it thinks
it's measuring -- this bit the router and the tile before REQUIRED_DEFINES
existed, see HANDOFF.md 1.6). This used to be tribal knowledge ("remember to
`` `ifndef `` a define before sweeping it"); this script makes it a build
failure instead.

    python3 p2/lint_defs.py           # exits 1 and lists violations if any
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFS_SVH = ROOT / "p2/rtl/sonic_defs.svh"


def guarded_defines(svh_text: str) -> set[str]:
    """Names defined inside an `ifndef NAME ... `define NAME ... `endif guard."""
    out = set()
    for m in re.finditer(r"`ifndef\s+(\w+)\b", svh_text):
        out.add(m.group(1))
    return out


def all_defines(svh_text: str) -> set[str]:
    return set(re.findall(r"`define\s+(\w+)\b", svh_text))


def overridden_macros() -> dict[str, set[str]]:
    """name -> set of places that override it via a `` `define `` / `-D`-style
    macro override (loop.py's REQUIRED_DEFINES, and each config.json's
    VERILOG_DEFINES). SYNTH_PARAMETERS is a *different*, always-safe mechanism
    -- it overrides a Verilog `parameter` directly via `-chparam`, with no
    macro/`` `ifndef `` involved, so it is intentionally not checked here."""
    sites: dict[str, set[str]] = {}

    loop_py = (ROOT / "p2/ppa/loop.py").read_text()
    m = re.search(r"REQUIRED_DEFINES\s*=\s*\{(.*?)\n\}", loop_py, re.S)
    if m:
        for name in re.findall(r'"(\w+)":\s*[\d.]+', m.group(1)):
            sites.setdefault(name, set()).add("p2/ppa/loop.py:REQUIRED_DEFINES")

    for cfg in sorted(ROOT.glob("p4/openlane/*/config*.json")):
        try:
            data = json.loads(cfg.read_text())
        except json.JSONDecodeError:
            continue
        for entry in data.get("VERILOG_DEFINES", []):
            name = str(entry).split("=", 1)[0].strip()
            if name:
                sites.setdefault(name, set()).add(f"{cfg.relative_to(ROOT)}:VERILOG_DEFINES")

    return sites


def main() -> int:
    svh_text = DEFS_SVH.read_text()
    guarded = guarded_defines(svh_text)
    defined = all_defines(svh_text)
    sites = overridden_macros()

    # A macro absent from sonic_defs.svh entirely is fine -- it's a
    # required-external-define (loop.py's own docstring: "units that need a
    # define before they will elaborate standalone"), and there's nothing in
    # the file for an override to conflict with. The only real risk is a
    # macro that IS defined in sonic_defs.svh but WITHOUT a guard, where an
    # external `-D` override gets silently clobbered by the unconditional
    # `` `define `` when the file is included.
    violations = []
    for name, where in sorted(sites.items()):
        if name in defined and name not in guarded:
            violations.append((name, where, "defined WITHOUT an `ifndef guard -- "
                                              "an external override is silently ignored"))

    if violations:
        print(f"P2-13 lint: {len(violations)} unguarded/undefined parameter(s) "
              f"that some flow tries to override:\n")
        for name, where, reason in violations:
            print(f"  {name}: {reason}")
            for w in sorted(where):
                print(f"    overridden by {w}")
        return 1

    print(f"P2-13 lint: clean -- no unguarded macro override conflicts across "
          f"{len(sites)} externally-overridden name(s).")
    for name in sorted(sites):
        state = "guarded in sonic_defs.svh" if name in guarded else \
                "required-external (no default in sonic_defs.svh -- fine, must always be passed)"
        print(f"  {name}: {state}, overridden by {', '.join(sorted(sites[name]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
