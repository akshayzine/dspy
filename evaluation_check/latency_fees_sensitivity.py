#!/usr/bin/env python3
"""
print_di_with_tests.py

Strict pairing by EXACT filename match between two labels (DQN vs L1),
compute realised PnL per file as:
    realised_pnl = final_cash - initial_cash

Then print, for each window:
  window i (START -> END) : di
where di = realised_pnl(DQN) - realised_pnl(L1)

Finally print a summary with:
  - n, mean di, sd di (always)
  - If SciPy is available: paired t-test (t, df, one-/two-sided p), 95% CI, and Wilcoxon signed-rank (one-sided, median>0) p
  - If SciPy is NOT available: print a note to install SciPy for statistical analysis and skip those tests.

Paths:
  Base = <repo_root>/logs/<eval-subdir>/<label>/
  where <repo_root> = Path(__file__).resolve().parent.parent

Usage:
  python print_di_with_tests.py dqn_test l1_test
  python print_di_with_tests.py l1_test dqn_test --eval-subdir eval --initial-cash 1000
"""
from pathlib import Path
import argparse
import sys
import re
import math
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Try SciPy (preferred). If missing, we’ll still print mean/sd and a note to install SciPy.
try:
    import scipy.stats as st
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def detect_roles(label_a: str, label_b: str) -> Tuple[str, str]:
    la, lb = label_a.lower(), label_b.lower()
    is_dqn = lambda s: "dqn" in s
    is_l1  = lambda s: ("l1" in s) or ("l1" in s) or ("baseline" in s)
    if is_dqn(la) and is_l1(lb): return label_a, label_b
    if is_dqn(lb) and is_l1(la): return label_b, label_a
    if is_dqn(la) and not is_dqn(lb): return label_a, label_b
    if is_dqn(lb) and not is_dqn(la): return label_b, label_a
    return label_a, label_b  # fallback: first=DQN, second=L1


def base_logs_path(eval_subdir: str) -> Path:
    return Path(__file__).resolve().parent.parent / "logs" / eval_subdir


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", name)]


def name_to_single_path(root: Path) -> Dict[str, Path]:
    """Map filename -> first matching Path found recursively under root (warn on duplicates)."""
    mapping: Dict[str, Path] = {}
    dups: Dict[str, int] = {}
    for p in sorted(root.rglob("*.csv")):
        if p.name not in mapping:
            mapping[p.name] = p
        else:
            dups[p.name] = dups.get(p.name, 1) + 1
    if dups:
        print("[WARN] Duplicate filenames detected (using first occurrence):", file=sys.stderr)
        for fn, cnt in dups.items():
            print(f"  {fn} (count={cnt})", file=sys.stderr)
    return mapping


def pick_cash_col(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    if "cash" in lower: return lower["cash"]
    for alt in ("final_cash", "end_cash"):
        if alt in lower: return lower[alt]
    raise KeyError("No 'cash'/'final_cash'/'end_cash' column found.")


def last_non_null(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) == 0:
        raise ValueError("Cash column has no non-null values.")
    return float(s.iloc[-1])


# Two datetime tokens in filename → "YYYY-MM-DD_HH-MM-SS" or "YYYY-MM-DD_HH-MM"
DT_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}[_T]\d{2}-\d{2}-\d{2}"),
    re.compile(r"\d{4}-\d{2}-\d{2}[_T]\d{2}-\d{2}"),
]

def _normalize_dt_token(token: str) -> str:
    token = token.replace("T", "_")
    if "_" not in token: return token
    d, t = token.split("_", 1)
    return f"{d} {t.replace('-', ':')}"


def extract_window_label(fname: str) -> str:
    ordered = []
    seen = set()
    for m in re.finditer("|".join(p.pattern for p in DT_PATTERNS), fname):
        tok = m.group(0)
        if tok not in seen:
            seen.add(tok); ordered.append(tok)
    if len(ordered) >= 2:
        return f"{_normalize_dt_token(ordered[0])} -> {_normalize_dt_token(ordered[1])}"
    return fname


def mean_sd(vals: List[float]) -> Tuple[float, float]:
    n = len(vals)
    if n == 0:
        return float('nan'), float('nan')
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    return mu, sd


def main():
    ap = argparse.ArgumentParser(description="Print per-window di and summary stats (paired t-test + Wilcoxon).")
    ap.add_argument("label_a", help="First label (e.g., dqn_test or l1_test)")
    ap.add_argument("label_b", help="Second label (e.g., l1_test or dqn_test)")
    ap.add_argument("--eval-subdir", default="eval_logs", help="Eval subdir under logs (default: eval)")
    ap.add_argument("--initial-cash", type=float, default=1000.0, help="Initial cash for realised PnL (default: 1000.0)")
    args = ap.parse_args()

    dqn_label, l1_label = detect_roles(args.label_a, args.label_b)
    root = base_logs_path(args.eval_subdir)
    dqn_dir = root / dqn_label
    l1_dir  = root / l1_label

    # Directory checks
    if not dqn_dir.is_dir():
        print(f"[ERR] DQN directory not found: {dqn_dir}", file=sys.stderr); sys.exit(2)
    if not l1_dir.is_dir():
        print(f"[ERR] L1 directory not found: {l1_dir}", file=sys.stderr); sys.exit(2)

    dqn_map = name_to_single_path(dqn_dir)
    l1_map  = name_to_single_path(l1_dir)
    if not dqn_map:
        print(f"[ERR] No CSV files in DQN dir: {dqn_dir}", file=sys.stderr); sys.exit(2)
    if not l1_map:
        print(f"[ERR] No CSV files in L1 dir: {l1_dir}", file=sys.stderr); sys.exit(2)

    set_dqn, set_l1 = set(dqn_map.keys()), set(l1_map.keys())

    # Strict missing checks by exact filename
    missing_in_l1 = sorted(set_dqn - set_l1, key=natural_key)
    missing_in_dqn = sorted(set_l1 - set_dqn, key=natural_key)
    if missing_in_l1 or missing_in_dqn:
        for fn in missing_in_l1:
            print(f"[ERR] File '{fn}' present in DQN but NOT in L1.", file=sys.stderr)
        for fn in missing_in_dqn:
            print(f"[ERR] File '{fn}' present in L1 but NOT in DQN.", file=sys.stderr)
        sys.exit(2)

    common = sorted(set_dqn, key=natural_key)

    # ---- Header BEFORE the loop ----
    print(f"# Labels resolved: DQN='{dqn_label}', L1='{l1_label}'")
    # print(f"# realised_pnl = final_cash - initial_c ash (initial_cash = {args.initial_cash:.6f})".replace(" ", ""))
    # print("# di = realised_pnl(DQN) - realised_pnl(L1)")
    # print("#")
    # print("# Window (START -> END) : di")
    # --------------------------------

    diffs: List[float] = []
    for i, fname in enumerate(common, start=1):
        dqn_p, l1_p = dqn_map[fname], l1_map[fname]
        label = extract_window_label(fname)

        try:
            dqn_df = pd.read_csv(dqn_p); l1_df = pd.read_csv(l1_p)
            dqn_cash = last_non_null(dqn_df[pick_cash_col(dqn_df)])
            l1_cash  = last_non_null(l1_df[pick_cash_col(l1_df)])
        except Exception as e:
            print(f"[ERR] Problem with '{fname}': {e}", file=sys.stderr); sys.exit(2)

        dqn_realised = dqn_cash - args.initial_cash
        l1_realised  = l1_cash  - args.initial_cash
        di = dqn_realised - l1_realised
        diffs.append(di)

        # print(f"window {i} ({label}) : {di:.6f}")

    # ---- Summary stats ----
    diffs_np = np.asarray(diffs, dtype=float)
    n = int(diffs_np.size)
    mean_diff, sd_diff = mean_sd(diffs)
    print("\n# Summary")
    # print(f"n       = {n}")
    print(f"mean PnL difference (di) = {mean_diff:.6f}")

    # ---- Statistical tests (only if SciPy available) ----
    if not SCIPY_AVAILABLE:
        print("\n[INFO] SciPy not found. Install SciPy for statistical analysis (t-test, confidence intervals, Wilcoxon):")
        print("       pip install scipy")
        sys.exit(0)

    if n >= 2 and sd_diff > 0:
        se = sd_diff / math.sqrt(n)
        t_stat = mean_diff / se
        dfree = n - 1
        p_one = float(st.t.sf(t_stat, dfree))             # H1: mean_diff > 0
        p_two = float(st.t.sf(abs(t_stat), dfree) * 2.0)  # two-sided
        tcrit = float(st.t.ppf(0.975, dfree))
        ci_lo, ci_hi = mean_diff - tcrit * se, mean_diff + tcrit * se

        print(f"t-test  : p_value = {p_two:.6g}")
    else:
        print("t-test  : NA (need at least n>=2 and nonzero sd).")
        print("95% CI  : NA")
      


if __name__ == "__main__":
    main()
