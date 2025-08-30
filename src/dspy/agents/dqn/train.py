import os
import json
import csv
import copy
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
from dspy.agents.dqn.model import QNetwork
from dspy.agents.dqn.agent import DQNAgent
from dspy.agents.dqn.replay_buffer import make_replay_buffer
from dspy.agents.dqn.nstep_adder import NStepAdder
from dspy.utils import get_torch_device
import sys
from torch.amp import autocast, GradScaler


def train_dqn(train_config: dict, env_fn,run_config: dict,features_config: dict ):
    """
    Train a DQN agent in a SimEnv using experience replay and a target network.

    Args:
        train_config (dict): Configuration parameters for training.
        env_fn (callable): Function that returns an initialized SimEnv object.
    """

    # === Environment / agent / device ===
    env = env_fn()
    agent = env.agent
    device = get_torch_device(run_config.get("device"))  # 
    use_cuda = (device.type == "cuda")

    # Get the model from the agent and move it to the selected device

    
    model = agent.model.float().to(device)

    # Create target network and initialize it with the same weights as model
    # input_dim = model.net[0].in_features
    # output_dim = model.net[-1].out_features
    # target_model = type(model)(input_dim=input_dim, output_dim=output_dim).float().to(device)
    # target_model.load_state_dict(model.state_dict())
    # target_model.eval()

    input_dim  = getattr(model, "input_dim", env.state_dim)
    output_dim = getattr(model, "output_dim", agent.num_actions if hasattr(agent, "num_actions") else None)
    if output_dim is None:
        output_dim = len(agent.action_set)  # fallback if you keep action_set

    # Make identical target by deep-copying the online net (handles all kwargs)
    target_model = copy.deepcopy(model).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()                  # turns off Dropout/BN
    for p in target_model.parameters():
        p.requires_grad_(False)
    



    # Load pretrained model weights if a path is provided
    if train_config.get("load_path"):
        model.load_state_dict(torch.load(train_config["load_path"]))
        target_model.load_state_dict(model.state_dict())

    # Optimizer setup
    
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config["lr"],
            weight_decay=train_config.get("weight_decay", 0.0),
            betas=(0.9, 0.999)  ###added
        )

    # Replay buffer for experience replay
    # buffer = ReplayBuffer(train_config["buffer_size"], pin_memory=use_cuda)
    
    replay = make_replay_buffer(device,
                            state_dim=env.state_dim,
                            capacity=train_config["replay_capacity"],
                            )
    # === AMP only on CUDA ===
    use_amp = (device.type == "cuda") and train_config.get("amp", True)
    scaler  = GradScaler("cuda", enabled=use_amp) if use_amp else None



    # Discount factor for Bellman equation
    gamma = train_config["gamma"]
    n_reward_step = train_config["n_reward_step"]
    gamma_n = gamma**(n_reward_step)
    adder = NStepAdder(n=n_reward_step, gamma=gamma)

    # Epsilon-greedy exploration parameters

    # hparams once
    epsilon_start  = float(train_config["epsilon_start"])
    epsilon_end    = float(train_config["epsilon_end"])
    epsilon_warmup = int(train_config.get("epsilon_warmup_ticks", 0))
    steps_per_episode = int(env.n_steps)                # must be constant
    num_episodes = int(train_config["num_episodes"])
    epsi = epsilon_start
    agent.epsilon = epsi

    # Other hyperparameters
    sync_interval = train_config["sync_interval"]
    batch_size = train_config["batch_size"]
    max_grad_norm = train_config.get("max_grad_norm", None)
    train_freq = train_config["train_freq"]
    min_replay = int(train_config.get("min_replay", 50_000))
    use_huber  = bool(train_config.get("use_huber", True))
    double_dqn = bool(train_config.get("double_dqn", True))

    # Base directory for saving logs and models
    # Create a single timestamped folder only once
    timestamp_f = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(__file__).parent.parent.parent.parent.parent / "logs/train_logs/dqn" / timestamp_f
    log_dir.mkdir(parents=True, exist_ok=True)
    episode_logging = train_config.get("episode_logging", True)
    step_logging = train_config.get("step_logging", False)
    log_file = "training_logs.csv"


     # === Train ===
    global_step = 0  # Track total steps across all episodes
    for episode in range(train_config["num_episodes"]):
        env.reset_state()
        adder.reset_window()
        total_loss = 0.0
        step_count = 0
        reward_sum = 0.0
        abs_reward_sum = 0.0
        action_counter = np.zeros(output_dim)
        prev_state = None   
        prev_action = None
        done = env.is_done()
        while not done :
            # print(global_step,"==")
            # === Get current environment state ===
            i = env.ptr  # Current index in the LOB data
            env.pre_step()  # Pre-step 
            state = env.get_state_vector(i)
            done = env.is_done()
            # print(f"Episode: {episode}, Step: {env.ptr}, Done: {done}, Data Length: {len(env.book)}")

            # Get basic LOB snapshot (best ask and best bid)
            best_ask = float(env.best_ask(i))
            best_bid = float(env.best_bid(i))

            # Sync agent inventory with environment
            agent.inventory = env.inventory

            # === Select action using epsilon-greedy policy ===
            action = agent.act(state, explore=True)
            agent.set_action_idx(action)
            action_counter[action] +=1
        
            # === Set quotes and simulate one environment step ===
            quotes = agent.get_quotes(state, best_ask, best_bid)
            env.inject_quotes(
                    quotes["bid_px"],
                    quotes["bid_qty"],
                    quotes["ask_px"],
                    quotes["ask_qty"],
                )
            env.step_with_injected_quotes()

            # done_check = env.is_done() #for end of episode check
            
            # === Store transition in replay buffer ===
            # if not done_check:
            #     next_state = env.get_state_vector(i)
            if done:
                last_idx = max(env.ptr - 1, 0)
                if env.inventory != 0:
                    env.square_off(last_idx)
                reward = env.reward
                reward_sum += reward
                abs_reward_sum += abs(reward)
                if prev_state is not None and prev_action is not None:
                    # replay.add(prev_state, int(prev_action), np.float32(reward), state, 1 if done else 0)
                    out = adder.push(prev_state.copy(), int(prev_action), reward, state.copy(), bool(done))
                    if out is not None:
                        s0, a0, Rn, sN, dN = out
                        replay.add(s0, a0, Rn, sN, dN)

                #Final Steps        
                for (s0, a0, Rtail, sTail, dTail) in adder.flush_end_episode():
                    replay.add(s0, a0, Rtail, sTail, dTail)
            else:   
                reward = env.reward
                reward_sum += reward
                abs_reward_sum += abs(reward)

                if prev_state is not None and prev_action is not None:
                    # replay.add(prev_state, int(prev_action), np.float32(reward), state, 1 if done else 0)
                    out = adder.push(prev_state.copy(), int(prev_action), reward, state.copy(), bool(done))
                    if out is not None:
                        s0, a0, Rn, sN, dN = out
                        replay.add(s0, a0, Rn, sN, dN)
                    

            
            # === Store the last transition ===
            prev_state = state
            prev_action = action
           
            # === Log step data if enabled ===
            if step_logging:
                mid = (best_ask + best_bid)*0.5
                step_data = {
                    "episode": episode,
                    "step": env.ptr,
                    "action_idx": action,
                    "bid_px": quotes["bid_px"],
                    "ask_px": quotes["ask_px"],
                    "bid_qty": quotes["bid_qty"],
                    "ask_qty": quotes["ask_qty"],
                    "inventory": env.inventory,
                    "cash": env.cash,
                    "reward": reward,
                    "mid": mid,
                    "cumulative_pnl": env.total_pnl(mid),
                    "num_trades": env.trade_count,
                    "done": int(done)
                }
                # Write to CSV log file
                step_filename = f"step_logs_episode_{episode}.csv"
                csv_writer(
                    log_dir=str(log_dir),
                    filename=step_filename,
                    log_data=step_data)  
            
            
            
            step_count += 1
            global_step += 1
            # === Train model if enough samples in buffer ===
            if (len(replay) >= max(batch_size, min_replay)) and (global_step % train_freq == 0):
                
                pin = use_cuda
                
                s, a, r, s_, d = replay.sample(batch_size)

                # If on CUDA, pin the sampled views so non_blocking H2D is actually async
                if use_cuda:
                    s, a, r, s_, d = (t.pin_memory() for t in (s, a, r, s_, d))



                 # Move to device with async H2D if pinned
                s  = s.to(device, dtype=torch.float32, non_blocking=pin)
                a  = a.to(device, dtype=torch.long, non_blocking=pin).unsqueeze(1)
                r  = r.to(device, non_blocking=pin)
                s_ = s_.to(device, dtype=torch.float32, non_blocking=pin)
                d  = d.to(device, dtype=torch.float32, non_blocking=pin)

                optimizer.zero_grad(set_to_none=True)

                # === Forward / loss / backward (AMP on CUDA) ===
                if use_amp:
                    with autocast(device_type="cuda", dtype=torch.float16, enabled=True): # RTX 6000 (Turing) → FP16 autocast
                        q_val = model(s).gather(1, a).squeeze(1)  # Q(s,a)

                        with torch.no_grad():
                            if double_dqn:
                                a_star = model(s_).argmax(dim=1, keepdim=True)
                                next_q = target_model(s_).gather(1, a_star).squeeze(1)
                            else:
                                next_q = target_model(s_).max(dim=1).values

                            target = (r + gamma_n * next_q * (1.0 - d)).float() #Ensure FP32

                        loss = F.smooth_l1_loss(q_val, target) if use_huber else F.mse_loss(q_val, target)

                    scaler.scale(loss).backward()
                    if max_grad_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    # Plain FP32 path (CPU or CUDA without AMP)
                    q_val = model(s).gather(1, a).squeeze(1)

                    with torch.no_grad():
                        if double_dqn:
                            a_star = model(s_).argmax(dim=1, keepdim=True)
                            next_q = target_model(s_).gather(1, a_star).squeeze(1)
                        else:
                            next_q = target_model(s_).max(dim=1).values

                        target = r + gamma_n * next_q * (1.0 - d)

                    loss = F.smooth_l1_loss(q_val, target) if use_huber else F.mse_loss(q_val, target)
                    loss.backward()
                    if max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                total_loss += float(loss.item())

                # === Periodically sync target network with online network ===
                # if global_step % sync_interval == 0:
                #     target_model.load_state_dict(model.state_dict())
                # === Soft-sync target network (polyak) ===
                with torch.no_grad():
                    tau = 0.002   # (try 0.002–0.005)
                    for pt, p in zip(target_model.parameters(), model.parameters()):
                        pt.mul_(1 - tau).add_(tau * p)

            # === Epsilon update ===
            epsi =epsilon_linear_by_global_tick(
                    episode=episode,
                    env_ptr=i,
                    steps_per_episode=steps_per_episode,
                    num_episodes=num_episodes,
                    min_replay=min_replay,
                    eps_start=epsilon_start,
                    eps_end=epsilon_end,
                    eps_warmup_ticks=epsilon_warmup,
                )
            agent.epsilon = epsi

        

        # === Logging ===
        avg_loss = total_loss / max(step_count, 1)
        avg_reward = reward_sum / step_count if step_count > 0 else 0
        avg_abs_reward = abs_reward_sum / step_count if step_count > 0 else 0
        max_action = agent.action_set[np.argmax(action_counter)]
        print(f"[Episode {episode}] Inventory: {env.inventory:.2f}, Cash: {env.cash:.2f}, Avg Loss: {avg_loss:.6f}, Epsilon: {epsi:.4f}")
        print(f"               Reward Sum: {reward_sum:.4f}, Avg Reward: {avg_reward:.4f}, Abs Avg Reward: {avg_abs_reward:.4f}")       
        if episode_logging:
            # Log data dictionary (after episode finishes)
            last_idx = max(env.ptr - 1, 0)
            last_mid = (env.best_ask(last_idx) + env.best_bid(last_idx))*0.5
            log_data = {
                "episode": episode,
                "avg_loss": avg_loss,
                "final_inventory": env.inventory,
                "final_cash": env.cash,
                "total_cost": env.total_cost,
                "realized_pnl": env.total_pnl(last_mid),
                "drawdown":env.drawdown,
                "max_drawdown":env.max_drawdown,
                "num_steps": step_count,
                "num_trades": env.trade_count,
                "zero_reward_count":env.zero_reward_count,
                "reward_sum": reward_sum,
                "abs_reward_sum": abs_reward_sum,
                "avg_reward": avg_reward,
                "avg_abs_reward": avg_abs_reward,
                "max_action":max_action,
                "max_action_count":np.max(action_counter),
                "epsilon": epsi
            }
            
            # Log it to a file like: src/dspy/agents/dqn/saved/logs_2025-10-06.csv
            
            csv_writer(
                log_dir=str(log_dir),
                filename=log_file,
                log_data=log_data
            )


    # === Save final model if path is specified ===
    # if train_config.get("save_path"):  # still use this as a toggle
    save_dir = Path(__file__).parent.parent / "dqn"/"saved/"/ timestamp_f
    save_dir.mkdir(parents=True, exist_ok=True)
    
    #Save is src/agents/dqn/saved/ folder with timestamp
    save_model_and_config(model, train_config,run_config, features_config, env,base_dir=save_dir)

    #Save in Log dir as well
    save_model_and_config(model, train_config,run_config, features_config, env,base_dir=log_dir)


def save_model_and_config(model, train_config,run_config, feature_config, env, base_dir):
    """
    Saves model and training config in a timestamped folder.

    Args:
        model        : Trained PyTorch model (QNetwork)
        train_config : Dict containing training hyperparameters
        env          : Environment object containing start_time and end_time attributes
        base_dir     : Base directory for saving artifacts
    """
    # now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # # start_ts = env.start_time.strftime("%Y%m%d-%H%M%S")
    # # end_ts   = env.end_time.strftime("%Y%m%d-%H%M%S")
    # base_dir = os.path.join(Path(__file__).parent.parent,base_dir)
    # print(base_dir)
    # folder_name = f"{now}"
    # save_path = os.path.join(base_dir, folder_name)
    # os.makedirs(save_path, exist_ok=True)
    save_path = base_dir

    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

    # Save training config
    with open(os.path.join(save_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)
    
    # Save run config
    with open(os.path.join(save_path, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=4)

     # Save training config
    with open(os.path.join(save_path, "feature_config.json"), "w") as f:
        json.dump(feature_config, f, indent=4)
    print(f"\n Model and config saved to: {save_path}")


def csv_writer(log_dir: str, filename: str, log_data: dict):
    """
    Logs per-episode training metrics to a CSV file.

    Args:
        log_dir (str): Directory to store the CSV log file.
        filename (str): Name of the CSV file (e.g., "training_log.csv").
        log_data (dict): Dictionary containing episode stats (one row).
    """
    log_path = Path(log_dir) / filename
    log_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # Write header if file does not exist
    write_header = not log_path.exists()

    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(log_data)

def epsilon_linear_by_global_tick(
    episode: int,
    env_ptr: int,
    steps_per_episode: int,
    num_episodes: int,
    min_replay: int,
    eps_start: float,
    eps_end: float,
    eps_warmup_ticks: int = 0,
) -> float:
    """
    Smooth ε schedule based on *global ticks* assuming constant episode length.

    ε = eps_start + (eps_end - eps_start) * progress
    where progress uses total_ticks = episode*steps_per_episode + env_ptr

    Args:
        episode:          current episode index (0-based)
        env_ptr:          current tick pointer within the episode (0..steps_per_episode-1)
        steps_per_episode:env.n_steps (must be constant across episodes)
        num_episodes:     planned number of training episodes
        eps_start:        starting epsilon
        eps_end:          ending epsilon (floor)
        eps_warmup_ticks: keep ε fixed at eps_start for this many initial ticks (optional)
    """
    total_ticks = episode * steps_per_episode + env_ptr
    total_planned = steps_per_episode * num_episodes
    if total_ticks < min_replay:
        return eps_start
    num = max(0, total_ticks - eps_warmup_ticks)
    den = max(1, total_planned - eps_warmup_ticks)
    prog = min(1.0, num / den)

    return float(eps_start + (eps_end - eps_start) * prog)
