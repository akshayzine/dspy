from collections import deque
from typing import Deque, Optional, Tuple, Iterable
import numpy as np

class NStepAdder:
    """
    O(n) n-step adder with optional time-aware discounting (uses timestamps).

    Emits (when ready): (s0, a0, R_n, s_n, d_n, gamma_prod)
      - R_n      : n-step discounted return over the window
                   If time-aware:    R_n = r1 + g1 r2 + g1 g2 r3 + ...
                   where g_i = (gamma_nano) ** Δt_i
                   If constant-γ:    R_n = r1 + γ r2 + γ^2 r3 + ...
      - gamma_prod: ∏ of step discounts across the window (0.0 for terminal tails)

    Minimal change from your prior adder:
      - push(...) now takes an extra `time_curr` (the tape timestamp for this step)
      - pass `gamma_nano` at construction to enable time-aware discounting

    """

    __slots__ = (
        "n", "gamma", "gamma_nano", "ts_unit", "dt_cap", "use_time",
        "_ts_scale", "S", "A", "R", "S2", "D", "T"
    )

    def __init__(
        self,
        n: int,
        gamma: float,
        gamma_nano: float | None = None,
        dt_cap: float = 3*1e9 
    ):
        """
        Args:
            n            : n-step window length (>=1)
            gamma        : per-event discount (used if gamma_nano is None)
            gamma_nano   : per-nano_second discount; if set, use γ_eff = γ_nanos ** Δt
            dt_cap       : cap Δt (nanos) to avoid huge gaps/outliers
        """
        assert n >= 1, "n-step must be >= 1"
        self.n = int(n)
        self.gamma = float(gamma)
        self.gamma_nano = float(gamma_nano) if gamma_nano is not None else None
        self.dt_cap = float(dt_cap)
        self.use_time = self.gamma_nano is not None


        # Deques (oldest -> newest)
        self.S: Deque = deque()
        self.A: Deque = deque()
        self.R: Deque = deque()
        self.S2: Deque = deque()
        self.D: Deque = deque()
        self.T: Deque = deque()  # timestamps aligned with rewards/next-states

    def reset_window(self) -> None:
        self.S.clear(); self.A.clear(); self.R.clear()
        self.S2.clear(); self.D.clear(); self.T.clear()

    # ---- core math ----
    def _discounted_sum_from_times(
        self, rs: Iterable[float], ts: Iterable[float]
    ) -> tuple[float, float]:
        """
        Variable-step discount:
          R_n = r1 + g1 r2 + g1 g2 r3 + ... ; g_i = γ_per_sec^{Δt_i} (or γ if constant)
        Returns (R_n, gamma_prod). Works directly over deques/iterables (no list()).
        """
        it_r = iter(rs)
        it_t = iter(ts)

        try:
            r0 = float(next(it_r))
            t_prev = float(next(it_t))
        except StopIteration:
            return 0.0, 0.0

        acc = r0
        gprod = 1.0

        if self.use_time:
            gps = self.gamma_nano
            dcap = self.dt_cap
            for r, t in zip(it_r, it_t):
                dt = (float(t) - t_prev) # in nano
                if dt < 0.0: dt = 0.0
                if dt > dcap: dt = dcap
                g_step = gps ** dt
                gprod *= g_step
                acc += gprod * float(r)
                t_prev = float(t)
        else:
            g = self.gamma
            for r, _t in zip(it_r, it_t):
                acc += gprod * float(r)
                gprod *= g
                
            gprod /= g  # 

        return float(acc), float(gprod)

    # ---- public API ----
    def push(self, s, a, r, s_next, done_next, time_curr) -> Optional[Tuple]:
        """
        Append one 1-step transition (timestamp = time_curr) and maybe emit an n-step tuple.

        Args:
            s, a, r, s_next, done_next : usual 1-step pieces (same tick)
            time_curr                  : current *tape* timestamp (units = ts_unit)

        Returns:
            None OR (s0, a0, R_n, s_n, d_n, gamma_prod)
        """
        # append newest step
        self.S.append(s); self.A.append(int(a)); self.R.append(float(r))
        self.S2.append(s_next); self.D.append(bool(done_next)); self.T.append(time_curr)

        # Emit when window reached n and newest step is non-terminal
        if (not done_next) and (len(self.R) == self.n):
            Rn, gamma_prod = self._discounted_sum_from_times(self.R, self.T)
            out = (
                self.S[0],                 # s0
                self.A[0],                 # a0
                np.float32(Rn),            # R_n
                self.S2[-1],               # s_n
                np.float32(0.0),           # d_n
                np.float32(gamma_prod),    # ∏ step discounts
            )
            # slide window by one
            self.S.popleft(); self.A.popleft(); self.R.popleft()
            self.S2.popleft(); self.D.popleft(); self.T.popleft()
            return out

        return None

    def flush_end_episode(self):
        """
        Emit all remaining windows as terminal (d_n=1.0; gamma_prod=0), then reset.
        """
        outs = []
        while len(self.R) > 0:
            Rn, _ = self._discounted_sum_from_times(self.R, self.T)
            outs.append((
                self.S[0],               # s0
                self.A[0],               # a0
                np.float32(Rn),          # R_L  (L <= n)
                self.S2[-1],             # s_L
                np.float32(1.0),         # d_n = 1.0 (terminal)
                np.float32(0.0),         # gamma_prod = 0.0 (no bootstrap)
            ))
            # slide by one to expose the next tail
            self.S.popleft(); self.A.popleft(); self.R.popleft()
            self.S2.popleft(); self.D.popleft(); self.T.popleft()

        self.reset_window()
        return outs
