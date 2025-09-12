# DSPy — RL Market‑Making Simulator & LOB Replay

> A modular Python framework for **high‑frequency market making research** on Binance BTCUSDT via **Tardis** LOB replay.
> Core pieces: event‑driven `MarketSimulator`, feature registry (Polars), and a DQN agent with replay & n‑step returns.

## At a glance
- **Data loader**: **Tardis only** (`dspy.hdb.tardis_*`).
- **Simulator**: event‑driven LOB replay with outbound latency, at‑touch fills, inventory/cost accounting, reward = realized cash – inventory penalty.
- **Agents**: DQN (`src/dspy/agents/dqn/*`) with Double DQN, Dueling head, n‑step targets, ε‑anneal, AMP; baseline `symmetric_l1`.
- **Config‑first runs**: Edit JSONs under `run/` then `python run/run_simulation.py`.
- **Outputs**: logs under `logs/`, checkpoints under `src/dspy/agents/dqn/saved/`.

---

## Project structure (essentials)
```
dspy/
├─ pyproject.toml
├─ README.md
├─ env/
│  └─ requirements.txt            # <-- install everything listed here
├─ data/
│  └─ tardis/
│     └─ processed/               # <-- put processed parquet here
│        ├─ binance-futures_book_YYYYMMDD_BTCUSDT.parquet
│        └─ binance-futures_trades_YYYYMMDD_BTCUSDT.parquet
├─ logs/
│  ├─ train_logs/
│  │  └─ dqn/                     # training logs (+ optional copy of checkpoints)
│  └─ eval_logs/                  # evaluation CSVs
├─ run/
│  ├─ features.json               # feature recipe & scaling
│  ├─ run_config.json             # product/sim settings + agent selector
│  └─ train_dqn_config.json       # DQN hyper‑parameters
├─ scripts/
│  ├─ run_sim_cpu.sbatch          # Warwick Avon CPU example
│  └─ run_sim_gpu.sbatch          # Warwick Avon GPU example
└─ src/dspy/
   ├─ hdb/                        # Tardis loader/registry
   ├─ features/                   # feature registry & utils
   ├─ sim/                        # MarketSimulator, Order, SimEnv wrapper
   └─ agents/
      ├─ symmetric_l1/            # baseline at‑touch L1 agent
      └─ dqn/
         ├─ agent.py, model.py, train.py, replay_buffer.py, nstep_adder.py
         └─ saved/
            ├─ run_model/model.pt # <-- default checkpoint path used for eval
            └─ YYYY-MM-DD_HH-MM-SS/{model.pt, run_config.json, train_config.json, feature_config.json}
```


---

## Installation
```bash
# 1) Clone & create a Python >=3.12 venv
git clone <your-repo-url> && cd dspy
python -m venv .venv && source .venv/bin/activate

# 2) Install the pinned dependencies from env/*.txt
#    Use the actual filename in your repo, e.g.:
pip install -r env/requirements.txt

# 4) Install data/tooling extras (no Bybit)
#    Choose ONE torch build: CPU or CUDA (find the right URL for your CUDA version).
pip install tardis-dev
# CPU‑only PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Example CUDA 12.x (adjust as needed):
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Data layout (processed only)
Place **processed parquet** under:
```
dspy/data/tardis/processed/
  ├─ binance-futures_book_YYYYMMDD_BTCUSDT.parquet
  └─ binance-futures_trades_YYYYMMDD_BTCUSDT.parquet
```

---

## Config cheat‑sheet (annotated)
> Comments (`// ...`) are for explanation—**remove** them in actual JSON.

**`run/run_config.json`**
```jsonc
{
  "dataset": "tardis",                 // fixed: Tardis loader
  "product": "BTCUSDT",                // instrument symbol
  "market": "binance-futures",         // venue/stream label (used by loader)
  "depth": 5,                          // LOB levels to replay (near‑touch features)
  "tick_size": 0.10,                   // price grid (sim clamps to this)
  "latency_micros": 10000,             // outbound order latency in microseconds
  "cost_in_bps": 0.2,                  // taker/maker fee model in basis points (fills only)
  "max_inventory": 0.0001,             // absolute inventory cap
  "inventory_penalty": 0.025,          // λ for quadratic inventory penalty in reward
  "initial_cash": 100000,              // starting cash (used for realized PnL)
  "min_order_size": 0.0001,            // minimum size
  "device": "auto",                    // "cpu" | "cuda" | "auto"
  "label": "training_dqn_run1",        // short tag used in logs/paths
  "agent": { "type": "dqn", "mode": "train" },   // DQN training; use "pretrained" for eval
  "simulator_mode": "train",           // "train" or "eval"
  "standard_scaling_feat": true,       // apply stored mean/std from features config
  "intervals": [                       // time windows (inclusive/exclusive by loader)
    { "start": "2025-04-11 00:29:00", "end": "2025-04-14 22:29:30" }
  ]
}
```

**`run/train_dqn_config.json`**
```jsonc
{
  "n_reward_step": 3,                  // n‑step return horizon (bias/variance trade‑off)
  "time_based_discount": true,         // if true, use Δt‑aware discounting
  "gamma": 0.997,                      // ONLY used when time_based_discount == false
  "gamma_half_life_sec": 5.0,          // if time‑based, per‑second decay s.t. γ^(half_life)=0.5
  "min_replay": 5000,                  // steps before starting SGD
  "batch_size": 256,                   // transitions per SGD update
  "train_freq": 10,                     // environment steps per SGD update
  "sync_interval": 2000,               // target network sync period (in SGD updates or steps)
  "double_dqn": true,                  // Double Q‑learning for overestimation bias reduction
  "dueling": true,                     // Dueling head (value + advantage decomposition)
  "epsilon_start": 1.0,                // ε‑greedy: starting exploration
  "epsilon_end": 0.02,                 // ε‑greedy: final exploration
  "epsilon_warmup_ticks": 10000,      // ticks to anneal ε from start→end
  "use_huber": true,                   // Huber loss (robust to outliers)
  "max_grad_norm": 10.0,               // gradient clipping (stability)
  "amp": true                          // automatic mixed precision (throughput)
  "num_episodes": 50,                  // number of epochs over data
  "max_grad_norm": 8.0,                //gradient cutoff in backprop
  "episode_logging": true,             //log in every epsiode
  "step_logging": false,               //log every step (keep it false unless debug on small window)
}
```

---

## Training flow (DQN)
1. **Edit `run/run_config.json`** (set product, intervals, fees, latency, etc.).
   - **DQN (training)**: `"agent": {"type": "dqn", "mode": "train"}`
2. **Edit `run/train_dqn_config.json`** (n‑step, discounting, ε‑schedule, target sync, etc.).
3. **Run**
   ```bash
   cd dspy/run
   python run_simulation.py
   ```
4. **Artifacts**
   - Checkpoints: `src/dspy/agents/dqn/saved/YYYY-MM-DD_HH-MM-SS/{model.pt, *_config.json}`
   - Logs: `logs/train_logs/dqn/` (a copy of configs/checkpoint may be mirrored here).

---

## Evaluation flow
1. **Select the agent**
   - **Baseline**: `"agent": {"type": "symmetric_l1", "mode": "eval"}`
   - **DQN (pretrained)**: `"agent": {"type": "dqn", "mode": "pretrained"}`
2. **Point to a checkpoint**
   - Recommended: copy the trained model to
     ```
     src/dspy/agents/dqn/saved/run_model/model.pt
     ```
     (the evaluator looks here by default).
3. **Switch to eval**
   - In `run/run_config.json`: `"simulator_mode": "eval"` and set a clean test interval.
   - Keep `"standard_scaling_feat": true` so eval uses training scalers.
4. **Run**
   ```bash
   cd dspy/run
   python run_simulation.py
   ```
5. **Outputs**
   - Evaluation CSVs under `logs/eval_logs/` (PnL path, spread capture, trades, etc.).
   - Simulator squares off residual inventory on the last tick.

---

## Where things log / save (summary)
- **Training logs**: `dspy/logs/train_logs/dqn/`
- **Evaluation logs**: `dspy/logs/eval_logs/`
- **Checkpoints (canonical)**: `dspy/src/dspy/agents/dqn/saved/YYYY-MM-DD_HH-MM-SS/`
- **Default eval checkpoint**: `dspy/src/dspy/agents/dqn/saved/run_model/model.pt`

---

## License
See `LICENSE`.