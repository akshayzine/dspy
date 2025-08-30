from collections import deque
from typing import Deque, Optional, Tuple, Iterable
import numpy as np

class NStepAdder:
    """
    Safe O(n) n-step adder for DQN-style targets.

    What it emits (when ready):
        (s0, a0, R_n, s_n, d_n)
      where
        - s0, a0:   state/action at the *start* of the window,
        - R_n:      n-step discounted return over rewards in the window,
                    R_n = r_{t+1} + γ r_{t+2} + ... + γ^{n-1} r_{t+n}
        - s_n:      next-state at the *end* of the window,
        - d_n:      0.0 for non-terminal windows; 1.0 for tails at episode end
                    (so the learner naturally drops the bootstrap term).

    Usage pattern (every env step):
        out = adder.push(s_t, a_t, r_{t+1}, s_{t+1}, done_{t+1})
        if out is not None:
            replay.add(*out)

    At episode end:
        for out in adder.flush_end_episode():
            replay.add(*out)

    Complexity:
        - Per push: O(1) bookkeeping.
        - When ready to emit: O(n) to sum ≤ n rewards (n is small, e.g., 5).
        - No accumulators that can drift; numerically robust.
    """

    # __slots__ saves memory by fixing the instance attribute set.
    __slots__ = ("n", "gamma", "S", "A", "R", "S2", "D")

    def __init__(self, n: int, gamma: float):
        """
        Args:
            n (int):      the n in "n-step" (window length).
            gamma (float):discount factor γ in (0, 1].

        Internal buffers (all aligned oldest -> newest):
            S  : deque of states s_t          (for each 1-step transition)
            A  : deque of actions a_t
            R  : deque of rewards r_{t+1}     (1-step rewards)
            S2 : deque of next-states s_{t+1}
            D  : deque of done flags for s_{t+1} (True if terminal at s_{t+1})
        """
        assert n >= 1, "n-step must be >= 1"
        self.n = int(n)
        self.gamma = float(gamma)

        # We do not set maxlen on R; we explicitly control sliding.
        self.S: Deque = deque()
        self.A: Deque = deque()
        self.R: Deque = deque()
        self.S2: Deque = deque()
        self.D: Deque = deque()

    def reset_window(self) -> None:
        """
        Clear all deques. Call at episode reset, or after flush.
        """
        self.S.clear(); self.A.clear(); self.R.clear(); self.S2.clear(); self.D.clear()

    @staticmethod
    def _discounted_sum(rs: Iterable[float], gamma: float) -> float:
        """
        Compute sum_{k=0}^{L-1} gamma^k * rs[k] safely in float64.

        Args:
            rs    : iterable of rewards ordered oldest -> newest.
            gamma : discount factor γ.

        Returns:
            float: discounted sum as Python float (float64 precision).
        """
        acc = 0.0      # accumulator in float64
        g   = 1.0      # current power of γ, starts at γ^0
        for r in rs:
            acc += g * float(r)
            g   *= gamma
        return float(acc)

    def push(self, s, a, r, s_next, done_next) -> Optional[Tuple]:
        """
        Append a single 1-step transition and (maybe) emit one n-step tuple.

        Args (all correspond to the *same* tick):
            s         : s_t        (current state)
            a         : a_t        (action taken at s_t)
            r         : r_{t+1}    (reward observed *after* stepping)
            s_next    : s_{t+1}    (next state after stepping)
            done_next : done flag for s_{t+1} (episode terminates at s_{t+1})

        Returns:
            None OR a single 5-tuple (s0, a0, R_n, s_n, d_n) when the window
            reaches length n and the *current* transition is not terminal.
            (We emit tails with d_n=1.0 in flush_end_episode instead.)
        """
        # Append the 1-step pieces to the back (newest position)
        self.S.append(s)
        self.A.append(int(a))
        self.R.append(float(r))
        self.S2.append(s_next)
        self.D.append(bool(done_next))

        # Only emit when:
        #   - window just reached size n, AND
        #   - the current (newest) transition is NOT terminal.
        # If terminal, we wait and let flush_end_episode() mark tails with d_n=1.
        if (not done_next) and (len(self.R) == self.n):
            # Build n-step discounted return over the current window
            Rn = np.float32(self._discounted_sum(self.R, self.gamma))

            # Prepare output: start-of-window (oldest) to end-of-window (newest)
            out = (
                self.S[0],            # s0
                self.A[0],            # a0
                Rn,                   # R_n
                self.S2[-1],          # s_n  (next-state at window end)
                np.float32(0.0),      # d_n  (non-terminal window)
            )

            # Slide the window forward by one: drop the oldest transition
            self.S.popleft(); self.A.popleft(); self.R.popleft(); self.S2.popleft(); self.D.popleft()
            return out

        # Not ready to emit yet
        return None

    def flush_end_episode(self):
        """
        Emit *all* remaining windows as terminal (d_n = 1.0), then reset.

        Why terminal here?
            At episode end, there is no valid bootstrap beyond the last tick.
            So every tail window must be treated as done (d_n = 1.0), making the
            (1 - d_n) * γ^n * max_a' Q(s_n, a') term vanish in the target.

        Returns:
            list of (s0, a0, R_n, s_n, 1.0) tuples.
        """
        outs = []
        while len(self.R) > 0:
            # Discount over the *current* tail (length L <= n)
            Rn = np.float32(self._discounted_sum(self.R, self.gamma))
            outs.append((
                self.S[0],           # s0
                self.A[0],           # a0
                Rn,                  # R_L (shorter-than-n)
                self.S2[-1],         # s_L (end-of-tail next-state)
                np.float32(1.0),     # d_n = 1.0 (terminal)
            ))

            # Slide by one to emit the next shorter tail on the next loop iter
            self.S.popleft(); self.A.popleft(); self.R.popleft(); self.S2.popleft(); self.D.popleft()

        # Ready for the next episode
        self.reset_window()
        return outs
