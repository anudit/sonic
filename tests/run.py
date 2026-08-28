#!/usr/bin/env python3
"""Zero-dependency test runner. `pytest tests/` also works if you have it."""
import sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import test_spec  # noqa: E402

fns = [(n, f) for n, f in sorted(vars(test_spec).items())
       if n.startswith("test_") and callable(f)]
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
print(f"\n{len(fns) - len(fails)}/{len(fns)} passed")
sys.exit(1 if fails else 0)
