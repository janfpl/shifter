"""Standalone entry point for the out-of-process GPU probe.

Run as ``python -m shifter.registration._gpu_probe [--strategy NAME]``; prints a
single JSON line ``{"available": bool, "name": str, "reason": str}`` on stdout,
preceded by a ``##SHIFTER_CUDA_VERSIONS## {...}`` marker line.

``--strategy`` selects how the DLL search path is arranged (see
``gpu_utils._STRATEGIES``): ``isolated`` (default) drops system CUDA-toolkit dirs
from PATH so CuPy uses its bundled libraries; ``system`` injects the system CUDA
toolkit path; ``bundled`` leaves PATH untouched.

This is deliberately a separate module from :mod:`shifter.registration.gpu_utils`
(which the package ``__init__`` imports): running an already-imported module with
``python -m`` triggers a ``RuntimeWarning`` from :mod:`runpy` about re-execution.
Keeping the entry point here — untouched by the package ``__init__`` — avoids that.

The probe imports CuPy and JIT-compiles a test kernel through NVRTC, which can
fault natively on a mismatched CUDA install. Isolating it in this child process
means such a fault produces a non-zero exit code that the parent turns into
"GPU unavailable", instead of taking the application down. faulthandler is
enabled so a direct manual run of this module still prints the native stack.
"""

from __future__ import annotations

import faulthandler
import json
import sys


def _parse_strategy(argv: list[str]) -> str:
    if "--strategy" in argv:
        i = argv.index("--strategy")
        if i + 1 < len(argv):
            return argv[i + 1]
    return "isolated"


def main() -> None:
    faulthandler.enable()

    from shifter.registration.gpu_utils import _emit_cuda_version_marker, _probe_gpu

    strategy = _parse_strategy(sys.argv[1:])

    # Emit the detected CUDA version first, so the parent can report it even if
    # the NVRTC kernel compile below faults natively and this process dies.
    _emit_cuda_version_marker(strategy)

    available, name, reason = _probe_gpu(strategy=strategy)
    print(json.dumps({"available": available, "name": name, "reason": reason}))


if __name__ == "__main__":
    main()
