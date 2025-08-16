# dspy/agents/common/replay_buffer.py
from __future__ import annotations
import numpy as np
import torch

class NPReplayBuffer:
    """Fast on CPU: contiguous NumPy storage, zero-copy torch.from_numpy at sample."""
    def __init__(self, state_dim: int, capacity: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.S  = np.empty((self.capacity, self.state_dim), dtype=np.float32)
        self.A  = np.empty((self.capacity,),               dtype=np.int32)
        self.R  = np.empty((self.capacity,),               dtype=np.float32)
        self.S2 = np.empty((self.capacity, self.state_dim), dtype=np.float32)
        self.D  = np.empty((self.capacity,),               dtype=np.uint8)   # 0/1
        self.size = 0
        self.ptr  = 0

    def add(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, d01: float):
        self.S[self.ptr]  = s               # copies values (env can reuse buffer)
        self.A[self.ptr]  = int(a)
        self.R[self.ptr]  = np.float32(r)
        self.S2[self.ptr] = s2
        self.D[self.ptr]  = 1 if d01 >= 0.5 else 0
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self): return self.size

    def sample(self, batch_size: int, pin: bool = False):
        idx = np.random.randint(0, self.size, size=batch_size)
        s  = torch.from_numpy(self.S[idx])
        a  = torch.from_numpy(self.A[idx]).to(torch.long)
        r  = torch.from_numpy(self.R[idx])
        s2 = torch.from_numpy(self.S2[idx])
        d  = torch.from_numpy(self.D[idx].astype(np.float32, copy=False))
        if pin:  # no benefit on CPU-only runs; enable when sending to CUDA
            s = s.pin_memory(); a = a.pin_memory(); r = r.pin_memory()
            s2 = s2.pin_memory(); d = d.pin_memory()
        return s, a, r, s2, d


class TorchPinnedReplayBuffer:
    """Fastest on CUDA: CPU-pinned torch storage, async non_blocking H2D."""
    def __init__(self, state_dim: int, capacity: int, use_fp16_states: bool = False):
        self.cap = int(capacity); self.dim = int(state_dim)
        self.use_fp16 = bool(use_fp16_states)
        pm = dict(pin_memory=True)
        sdtype = torch.float16 if self.use_fp16 else torch.float32
        self.S  = torch.empty((self.cap, self.dim), dtype=sdtype, **pm)
        self.A  = torch.empty((self.cap,), dtype=torch.int32, **pm)
        self.R  = torch.empty((self.cap,), dtype=torch.float32, **pm)
        self.S2 = torch.empty((self.cap, self.dim), dtype=sdtype, **pm)
        self.D  = torch.empty((self.cap,), dtype=torch.uint8, **pm)
        self.size = 0
        self.ptr  = 0
        self._rng = torch.Generator(device='cpu')

    def add(self, s_np: np.ndarray, a: int, r: float, s2_np: np.ndarray, d01: float):
        # copy from NumPy into pinned tensors (env state is np.float32)
        s_t  = torch.from_numpy(s_np)    # CPU view
        s2_t = torch.from_numpy(s2_np)
        if self.use_fp16:
            self.S[self.ptr].copy_(s_t.to(torch.float16))
            self.S2[self.ptr].copy_(s2_t.to(torch.float16))
        else:
            self.S[self.ptr].copy_(s_t)          # fp32
            self.S2[self.ptr].copy_(s2_t)
        self.A[self.ptr] = int(a)
        self.R[self.ptr] = float(r)
        self.D[self.ptr] = 1 if d01 >= 0.5 else 0
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def __len__(self): return self.size

    def sample(self, batch_size: int):
        idx = torch.randint(self.size, (batch_size,), generator=self._rng)
        s  = self.S[idx]
        a  = self.A[idx].to(torch.long)
        r  = self.R[idx]
        s2 = self.S2[idx]
        d  = self.D[idx].to(torch.float32)
        return s, a, r, s2, d


def make_replay_buffer(device: torch.device, state_dim: int, capacity: int,
                       use_fp16_states_on_cuda: bool = False):
        if device.type == "cuda":
            return TorchPinnedReplayBuffer(state_dim, capacity, use_fp16_states_on_cuda)
        else:
            return NPReplayBuffer(state_dim, capacity)
