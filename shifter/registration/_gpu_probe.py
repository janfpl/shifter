"""Standalone entry point for the out-of-process GPU probe.

Run as ``python -m shifter.registration._gpu_probe``; prints a single JSON line
``{"available": bool, "name": str, "reason": str}`` on stdout.

This is deliberately a separate module from :mod:`shifter.registration.gpu_utils`
(which the package ``__init__`` imports): running an already-imported module with
``python -m`` triggers a ``RuntimeWarning`` from :mod:`runpy` about re-execution.
Keeping the entry point here — untouched by the package ``__init__`` — avoids that.

The probe imports CuPy and JIT-compiles a test kernel through NVRTC, which can
fault natively on a mismatched CUDA install. Isolating it in this child process
means such a fault produces a non-zero exit code that the parent turns into
"GPU unavailable", instead of taking the application down.
"""

from __future__ import annotations

import json


def main() -> None:
    from shifter.registration.gpu_utils import _probe_gpu

    available, name, reason = _probe_gpu()
    print(json.dumps({"available": available, "name": name, "reason": reason}))


if __name__ == "__main__":
    main()
