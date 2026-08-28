#!/usr/bin/env python3
"""Zero-dependency test runner. `pytest tests/` also works if you have it."""
import sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import test_spec  # noqa: E402

mods = [test_spec]
skipped = []
# The quantization kernels need torch. The headline promise is that every
# published number reproduces with numpy alone, so a missing torch skips this
# module rather than failing the run.
try:
    import test_quant  # noqa: E402
    mods.append(test_quant)
except ImportError as e:
    skipped.append(f"test_quant ({e.name} not installed)")

fns = [(n, f) for m in mods for n, f in sorted(vars(m).items())
       if n.startswith("test_") and callable(f) and f.__module__ == m.__name__]
fails = []
for name, fn in fns:
    try:
        fn()
        print(f"  ok   {name}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        if "-v" in sys.argv:
            traceback.print_exc()
        fails.append(name)
for s in skipped:
    print(f"  skip {s}")
print(f"\n{len(fns) - len(fails)}/{len(fns)} passed"
      + (f", {len(skipped)} module(s) skipped" if skipped else ""))
sys.exit(1 if fails else 0)
