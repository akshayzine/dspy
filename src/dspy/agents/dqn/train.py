import os
import json
import csv
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
from dspy.agents.dqn.model import QNetwork
from dspy.agents.dqn.agent import DQNAgent


class ReplayBuffer:
    def __init__(self, max_size=100_000):
        # Fixed-size buffer to store experience tuples
        self.buffer = deque(maxlen=max_size)

    def push(self, s, a, r, s_, done):
        # Add a single transition to the buffer
        self.buffer.append((s, a, r, s_, done))

    def sample(self, batch_size):
        # Sample a random batch of transitions
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s_, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32)
        )


def train_dqn(train_config: dict, env_fn,run_config: dict,features_config: dict ):
    """
    Train a DQN agent in a SimEnv using experience replay and a target network.

    Args:
        train_config (dict): Configuration parameters for training.
        env_fn (callable): Function that returns an initialized SimEnv object.
    """

    # Create environment and access the agent
    env = env_fn()
    agent = env.agent
    device = train_config["device"]

    # Get the model from the agent and move it to the selected device
    model = agent.model.to(device)

    # Create target network and initialize it with the same weights as model
    input_dim = model.net[0].in_features
    output_dim = model.net[-1].out_features
    target_model = type(model)(input_dim=input_dim, output_dim=output_dim).to(device)
    target_model.load_state_dict(model.state_dict())


    # Load pretrained model weights if a path is provided
    if train_config.get("load_path"):
        model.load_state_dict(torch.load(train_config["load_path"]))
        target_model.load_state_dict(model.state_dict())

    # Optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])

    # Replay buffer for experience replay
    buffer = ReplayBuffer(train_config["buffer_size"])

    # Discount factor for Bellman equation
    gamma = train_config["gamma"]

    # Epsilon-greedy exploration parameters
    epsilon = train_config["epsilon_start"]
    epsilon_end = train_config["epsilon_end"]
    epsilon_decay = train_config["epsilon_decay"]
    agent.epsilon = epsilon

    # Other hyperparameters
    sync_interval = train_config["sync_interval"]
    batch_size = train_config["batch_size"]
    max_grad_norm = train_config.get("max_grad_norm", None)

    # Base directory for saving logs and models
    # Create a single timestamped folder only once
    timestamp_f = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(__file__).parent.parent.parent.parent.parent / "logs/train_logs/dqn" / timestamp_f
    log_dir.mkdir(parents=True, exist_ok=True)
    episode_logging = train_config.get("episode_logging", True)
    step_logging = train_config.get("step_logging", True)

    # Define filename once
    log_file = "training_logs.csv"
    # Main training loop
    for episode in range(train_config["num_episodes"]):
        env.reset_state()
        total_loss = 0.0
        step_count = 0
        reward_sum = 0.0
        prev_state = None   
        prev_action = None

        while not env.is_done():
            
            # === Get current environment state ===
            state = env.get_state_vector()
            done = env.is_done()
            # print(f"Episode: {episode}, Step: {env.ptr}, Done: {done}, Data Length: {len(env.book)}")

            # Get basic LOB snapshot (best ask and best bid)
            lob_row = env.book[env.ptr]
            lob_state = [lob_row["asks[0].price"].item(), lob_row["bids[0].price"].item()]

            # Sync agent inventory with environment
            agent.inventory = env.inventory

            # === Select action using epsilon-greedy policy ===
            action = agent.act(state, explore=True)
            agent.set_action_idx(action)
        
            # === Set quotes and simulate one environment step ===
            quotes = agent.get_quotes(state, lob_state)
            env.inject_quotes({
                    "bid_px": quotes["bid_px"],
                    "bid_qty": quotes["bid_qty"],
                    "ask_px": quotes["ask_px"],
                    "ask_qty": quotes["ask_qty"],
                })

            env.step_with_injected_quotes()

            done_check = env.is_done() #for end of episode check
            
            # === Store transition in replay buffer ===
            if not done_check:
                next_state = env.get_state_vector()
        
            reward = env.reward
            reward_sum += reward
            if prev_state is not None and prev_action is not None:
                buffer.push(prev_state, prev_action, reward, state, done)
            
            # === Store the last transition ===
            prev_state = state
            prev_action = action
           
            # === Log step data if enabled ===
            if step_logging:
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
                    "mid": lob_row["mid"].item(),
                    "cumulative_pnl": env.total_pnl(mid=lob_row["mid"].item()),
                    "num_trades": env.trade_count,
                    "done": done
                }
                # Write to CSV log file
                step_filename = f"step_logs_episode_{episode}.csv"
                csv_writer(
                    log_dir=str(log_dir),
                    filename=step_filename,
                    log_data=step_data)  


            # === Train model if enough samples in buffer ===
            if len(buffer.buffer) >= batch_size:
                s, a, r, s_, d = buffer.sample(batch_size)

                # Move to device
                s = s.to(device)
                a = a.to(device).unsqueeze(1)  # Shape: [B, 1]
                r = r.to(device)
                s_ = s_.to(device)
                d = d.to(device)

                # === Q(s, a) prediction from online network ===
                q_vals = model(s)  # Shape: [B, num_actions]
                q_val = q_vals.gather(1, a).squeeze()  # Get Q-values of taken actions

                # === Compute Bellman target using target network ===
                with torch.no_grad():
                    next_q_vals = target_model(s_).max(1)[0]  # Max Q-value at next state
                    target = r + gamma * next_q_vals * (1 - d)  # Bellman equation

                # === Compute loss and backpropagate ===
                loss = F.mse_loss(q_val, target)
                total_loss += loss.item()
                step_count += 1

                optimizer.zero_grad()
                loss.backward()

                # Optional: gradient clipping
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

        # === Decay epsilon after every episode ===
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        agent.epsilon = epsilon

        # === Periodically sync target network with online network ===
        if episode % sync_interval == 0:
            target_model.load_state_dict(model.state_dict())

        # === Logging ===
        avg_loss = total_loss / max(step_count, 1)
        avg_reward = reward_sum / step_count if step_count > 0 else 0
        print(f"[Episode {episode}] Inventory: {env.inventory:.2f}, Cash: {env.cash:.2f}, Avg Loss: {avg_loss:.6f}, Epsilon: {epsilon:.4f}")

        if episode_logging:
            # Log data dictionary (after episode finishes)
            log_data = {
                "episode": episode,
                "avg_loss": avg_loss,
                "final_inventory": env.inventory,
                "final_cash": env.cash,
                "realized_pnl": env.total_pnl(mid=env.book[env.ptr - 1]["mid"].item()),
                "num_steps": step_count,
                "num_trades": env.trade_count,
                "reward_sum": reward_sum,
                "avg_reward": avg_reward,
                "epsilon": epsilon
            }
            
            # Log it to a file like: src/dspy/agents/dqn/saved/logs_2025-10-06.csv
            
            csv_writer(
                log_dir=str(log_dir),
                filename=log_file,
                log_data=log_data
            )


    # === Save final model if path is specified ===
    # if train_config.get("save_path"):  # still use this as a toggle
    save_dir = Path(__file__).parent.parent / "dqn/saved/"/ timestamp_f
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