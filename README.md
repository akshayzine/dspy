# RLMM(DSPy) — RL Market‑Making Simulator & LOB Replay

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
├─ src/dspy/
│  ├─ hdb/                        # Tardis loader/registry
│   ├─ features/                   # feature registry & utils
│   ├─ sim/                        # MarketSimulator, Order, SimEnv wrapper
│   └─ agents/
│      ├─ symmetric_l1/            # baseline at‑touch L1 agent
│      └─ dqn/
│         ├─ agent.py, model.py, train.py, replay_buffer.py, nstep_adder.py
│         └─ saved/
│            ├─ run_model/model.pt # <-- default checkpoint path used for eval
│            └─ YYYY-MM-DD_HH-MM-SS/{model.pt, run_config.json, train_config.json, feature_config.json}
└─ evaluation_check/    
   ├─ base_stats.py     
   ├─ base_latency_fees.json
   ├─ evaluation_check_base.sh  # Evaluation Results for base regime
   ├─ latency_fees_sensitivity.py
   ├─ sensitivity_latency_fees.json
   └─ evaluation_check_sensitivity.sh #Evaulation Results for latency fees sensitivity
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
  "dataset": "tardis",                     // fixed: Tardis loader
  "product": "BTCUSDT",                    // instrument symbol
  "market": "binance-futures",             // venue/stream label (used by loader)
  "depth": 5,                              // LOB levels to replay (near‑touch features)
  "tick_size": 0.10,                       // price grid (sim clamps to this)
  "latency_micros": 10000,                 // outbound order latency in microseconds
  "cost_in_bps": 0.2,                      // taker/maker fee model in basis points (fills only)
  "fixed_cost": 0,
  "max_inventory": 0.001,                  // absolute inventory cap
  "inventory_penalty": 0.04,               // λ for quadratic inventory penalty in reward
  "initial_cash": 1000,                    // starting cash (used for realized PnL)
  "min_order_size": 0.001,                 // minimum size
  "device": "auto",                        // "cpu" | "cuda" | "auto"
  "label": "eval_run",                     // short tag used in logs/paths
  "agent": { "type": "dqn", "mode": "eval" },   // DQN training; use "pretrained" for eval
  "simulator_mode": "eval",                // "train" or "eval"
  "standard_scaling_feat": true,           // apply stored mean/std from features config
  "eval_log_flag": true,                   //switch on/off logging 
  "intervals": [                           // time windows (inclusive/exclusive by loader)
    { "start": "2025-04-15 02:00:00", "end": "2025-04-15 05:00:00" }
  ]
}
```

**`run/train_dqn_config.json`**
```jsonc
{
  "lr": 1e-4,                         // learning rate
  "weight_decay": 1e-5,               // L2 weight decay
  "min_replay": 8000,                 // transitions required before training starts
  "mdp_step_warmup": 10000,           // initial ticks before enabling full MDP triggers
  "action_timeout_sec": 0.35,         // min dwell between actions during training
  "use_huber": true,                  // use Huber loss
  "double_dqn": true,                 // use Double DQN targets
  "n_reward_step": 1,                 // n-step return horizon
  "gamma": 0.99,                      // base per-step discount (used if time-based is off)
  "time_based_discount": true,        // use time-aware discount γ^{Δt}
  "gamma_half_life_sec": 5.0,         // if time-based, per-second decay s.t. γ^(half_life)=0.5
  "epsilon_start": 1.00,              // initial ε for ε-greedy
  "epsilon_end": 0.05,                // final ε for ε-greedy
  "epsilon_warmup_ticks": 10000,      // ticks to anneal ε from start→end
  "replay_capacity": 100000,          // max transitions stored in replay
  "batch_size": 512,                  // SGD batch size
  "sync_interval": 5000,              // target network sync interval (steps)
  "num_episodes": 50,                 // number of training episodes
  "max_grad_norm": 8.0,               // gradient clipping (L2 norm)
  "train_freq": 10,                   // env steps per gradient update
  "load_path": null,                  // optional checkpoint to load
  "save_path": null,                  // optional checkpoint save path
  "episode_logging": true,            // log per-episode metrics
  "step_logging": false,              // log per-step metrics
  "amp": true                         // use mixed precision
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

## Result Verification 
To verify statistical results, multiple simulations can be launched one by one, but convenience wrappers
are provided under `evaluation_check/`, which runs **both agents** on the evaluation windows using the pretrained DQN, logs results, and generates summaries from the logs:

Set values in `run/run_config.json`. Ensure the **evaluation intervals** are configured; set `"simulator_mode": "eval"` and `"eval_log_flag": true`.


### 1) Base regime comparison (10 ms latency; 0.20 bps fees; Section 7.4.1)
> **Note:** In the JSON, **latency is specified in microseconds**.

```bash
#go to evaluation_check folder
cd ../evaluation_check   
# verify latency/fees in: base_latency_fees.json   (latency in μs)
bash evaluation_check_base.sh
```
### 2) Latency–fee sensitivity (Section 7.4.2)
> **Note:** In the JSON grid, **latency is specified in microseconds**.

```bash
#go to evaluation_check folder
cd ../evaluation_check
# verify the grid in: sensitivity_latency_fees.json   (latency in μs)
bash evaluation_check_sensitivity.sh
```
---

## License
See `LICENSE`.