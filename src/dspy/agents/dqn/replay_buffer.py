# src/dspy/agents/dqn/replay_buffer.py
from __future__ import annotations
import torch
from typing import Optional, Tuple

class SimpleReplayBuffer:
    """
    Minimal replay buffer:
      - Stores transitions in preallocated CPU torch tensors.
      - No CUDA-specific branches.
      - sample(..., device=...) moves only the sampled batch to the training device.
    """
    def __init__(self, state_dim: int, capacity: int, state_dtype: torch.dtype = torch.float32):
        self.state_dim = int(state_dim)
        self.capacity  = int(capacity)
        self.S  = torch.empty((self.capacity, self.state_dim), dtype=state_dtype)   # [N, d]
        self.S2 = torch.empty((self.capacity, self.state_dim), dtype=state_dtype)   # [N, d]
        self.A  = torch.empty((self.capacity,), dtype=torch.long)                   # [N]
        self.R  = torch.empty((self.capacity,), dtype=torch.float32)                # [N]
        self.D  = torch.empty((self.capacity,), dtype=torch.bool)                   # [N]
        self.size = 0
        self.ptr  = 0

    def __len__(self) -> int:
        return self.size

    def add(self, s, a: int, r: float, s2, done) -> None:
        i = self.ptr
        s_t  = torch.as_tensor(s,  dtype=self.S.dtype,  device="cpu").contiguous()
        s2_t = torch.as_tensor(s2, dtype=self.S2.dtype, device="cpu").contiguous()
        self.S[i].copy_(s_t)
        self.S2[i].copy_(s2_t)
        self.A[i] = int(a)
        self.R[i] = float(r)
        self.D[i] = bool(float(done) >= 0.5)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.size > 0, "ReplayBuffer is empty."
        idx = torch.randint(self.size, (batch_size,), dtype=torch.int64)
        s  = self.S.index_select(0, idx)
        a  = self.A.index_select(0, idx)
        r  = self.R.index_select(0, idx)
        s2 = self.S2.index_select(0, idx)
        d  = self.D.index_select(0, idx).to(torch.float32)
        if device is not None:
            s  = s.to(device)
            a  = a.to(device)
            r  = r.to(device)
            s2 = s2.to(device)
            d  = d.to(device)
        return s, a, r, s2, d


# Keep the existing factory name so callers don't change
def make_replay_buffer(
    device: torch.device,  # kept for API compatibility; ignored here
    state_dim: int,
    capacity: int,
    use_fp16_states_on_cuda: bool = False,  # ignored in this simple version
) -> SimpleReplayBuffer:
    return SimpleReplayBuffer(state_dim=state_dim, capacity=capacity, state_dtype=torch.float32)
