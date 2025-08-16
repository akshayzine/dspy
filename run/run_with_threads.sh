#!/usr/bin/env bash
set -euo pipefail

# Move to this script's directory (so relative paths work)
cd "$(dirname "$0")"

# Threading/env knobs for Mac CPU runs (override by exporting before calling)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-6}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}

echo "=== Preflight ($(date)) ==="
echo "CWD: $(pwd)"
echo "Python: $(command -v python3)"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "POLARS_MAX_THREADS=$POLARS_MAX_THREADS"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "VECLIB_MAXIMUM_THREADS=$VECLIB_MAXIMUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"

# (Mac) CPU info
if command -v sysctl >/dev/null 2>&1; then
  echo "CPU: $(sysctl -n machdep.cpu.brand_string || true)"
  echo "Cores: physical $(sysctl -n hw.physicalcpu || true) / logical $(sysctl -n hw.logicalcpu || true)"
fi

# # Python-side thread/BLAS/polars/torch report (runs fast)
# python3 - <<'PY'
# import os, sys, platform, numpy as np
# print("\n--- Python/Libs ---")
# print(f"Python {sys.version.split()[0]} on {platform.platform()}")
# print(f"NumPy {np.__version__}")
# try:
#     np.__config__.show()
# except Exception:
#     pass
# try:
#     import polars as pl
#     print(f"Polars {pl.__version__}")
#     try:
#         print(f"Polars threadpool size: {pl.threadpool_size()}")
#     except Exception:
#         print("Polars threadpool size: n/a")
# except Exception as e:
#     print("Polars: not installed")
# try:
#     import torch
#     print(f"Torch {torch.__version__}")
#     print(f"Torch get_num_threads(): {torch.get_num_threads()}")
#     print(f"Torch get_num_interop_threads(): {torch.get_num_interop_threads()}")
# except Exception:
#     print("Torch: not installed")
# print("--- Env ---")
# for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS","POLARS_MAX_THREADS"]:
#     print(f"{k}={os.getenv(k)}")
# PY

echo "=== Launching ==="
exec python3 -u run_simulation.py "$@"
