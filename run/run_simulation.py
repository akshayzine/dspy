import json
import sys
from datetime import datetime
from pathlib import Path
import polars as pl
import os 
import torch


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dspy.hdb import get_dataset
from dspy.sim.market_simulator import MarketSimulator
from dspy.utils import to_ns, ts_to_str, get_torch_device
from dspy.features.feature_utils import apply_batch_features, extract_features , flatten_features
from dspy.agents.agent_utils import get_agent


# ---------- Load run config file ----------

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)




# ---------- Main simulation function ----------

def run_simulation(config: dict):
    dataset_name     = config["dataset"]
    product          = config["product"]
    depth            = config["depth"]
    latency_ns       = config["latency_micros"] * 1_000
    inv_penalty      = config["inventory_penalty"]/1e9  # Scale penalty to be per second
    initial_cash     = config["initial_cash"]
    agent_config     = config["agent"]
    intervals        = config["intervals"]
    min_order_size   = config["min_order_size"]
    max_inventory    = config["max_inventory"]
    tick_size        = config["tick_size"]
    initial_cash     = config["initial_cash"]
    cost_in_bps      = config["cost_in_bps"]
    fixed_cost       = config["fixed_cost"]
    simulator_mode   = config["simulator_mode"]
    eval_log_flag    = config["eval_log_flag"]
    std_flag         = config["standard_scaling_feat"]
    comp_system        = config["comp_system"]

    t_device = get_torch_device(config["device"])
    print("Using device:", t_device)
    # # Cap PyTorch threads when on CPU
    # if t_device.type == "cpu":
    #     _configure_torch_threads_cpu()

    loader = get_dataset(dataset_name)
    all_books, all_ts = [], []

    # ---------- Load cofeature config file ----------
    feature_path = Path(__file__).parent / "features.json"
    feature_config = load_config(feature_path)
    inventory_feature_flag = "inventory" in feature_config.keys()
    active_quotes_flag = "active_quotes" in feature_config.keys()
    pending_quotes_flag = "pending_quotes" in feature_config.keys()
    active_age_flag = 'active_order_age' in feature_config.keys()
    pending_age_flag = 'active_order_age' in feature_config.keys()
    
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print('Sim started at:', now)

    for interval in intervals:
        start_str = interval["start"]
        end_str   = interval["end"]

        start_ts = datetime.strptime(interval["start"], "%Y-%m-%d %H:%M:%S").strftime("%y%m%d.%H%M%S")
        end_ts   = datetime.strptime(interval["end"],   "%Y-%m-%d %H:%M:%S").strftime("%y%m%d.%H%M%S")
        # Load book data for the interval
        df = loader.load_book(
            product=product,
            times=[start_ts, end_ts],
            depth=depth,
            type="book_snapshot_25",
            lazy=False
        )
        print(len(df),"dfdsfasfs")
        # Get features from the book data
        if comp_system =='personal':
            num_chunks = 10
            chunk_size = len(df) // num_chunks # Use integer division
            processed_data_dir = Path(__file__).parent.parent/ "chunk_data"
            print(processed_data_dir)
            processed_data_dir.mkdir(parents=True, exist_ok=True)
            print(f"Splitting the DataFrame into {num_chunks} chunks of approximately {chunk_size} rows each.")

            # 3. Loop through each chunk, process it, and save it to disk
            
            for i in range(num_chunks):
                feature_cols = []
                print(f"--- Processing chunk {i+1}/{num_chunks} ---")
                
                # Calculate the start offset for the slice
                offset = i * chunk_size
                
                # Get the current chunk using a simple slice
                df_chunk = df.slice(offset, chunk_size)

                # Apply feature engineering to the chunk
                print(f"  - Applying batch features...")
                df_processed_chunk, feature_cols = apply_batch_features(df_chunk, feature_config,std_flag)
                feature_cols = [col for col in feature_cols if df_processed_chunk[col].dtype != pl.Datetime]

                # Save the processed chunk to disk
                output_path = processed_data_dir / f"processed_chunk_{i}.parquet"
                print(f"  - Saving processed chunk to {output_path}")
                df_processed_chunk.write_parquet(output_path)

            # 4. Release memory of the huge raw DataFrame
            print("\nReleasing memory of the original raw DataFrame...")
            del df
            gc.collect()


            # 5. Load all the processed chunks from disk
            print("--- Loading all processed chunks from disk ---")
            all_processed_files = list(processed_data_dir.glob("*.parquet"))
            # df = pl.read_parquet(all_processed_files)
            df = pl.scan_parquet(all_processed_files) 
        
        else:
            df, feature_cols = apply_batch_features(df, feature_config, std_flag)
            feature_cols = [col for col in feature_cols if df[col].dtype != pl.Datetime]

        #Drop mid price as feature
        # feature_cols = [col for col in feature_cols if col == 'mid']

        feature_length = (len(feature_cols)
                          + (1 if inventory_feature_flag else 0) + (4 if active_quotes_flag else 0) +(4 if pending_quotes_flag else 0)
                          +(2 if active_age_flag else 0) + (2 if pending_age_flag else 0) 
                          + (2 if active_age_flag or active_quotes_flag else 0) + (2 if pending_age_flag or pending_quotes_flag else 0 ))
        print(f"Feature length: {feature_length}")  
        # print(f"feature columns: {feature_cols}")
        #Get agent from the config
        agent = get_agent(config,feature_length)
        #Get simulator
        sim = MarketSimulator(
            book=df,
            feature_config=feature_config,
            feature_columns=feature_cols,
            inventory_feature_flag = inventory_feature_flag,
            active_quotes_flag=active_quotes_flag,
            pending_quotes_flag=pending_quotes_flag,
            active_age_flag = active_age_flag,
            pending_age_flag = pending_age_flag,
            agent=agent,
            latency_ns=latency_ns,
            inventory_penalty=inv_penalty,
            max_inventory=max_inventory,
            min_order_size=min_order_size,
            tick_size=tick_size,
            initial_cash=initial_cash,
            cost_in_bps=cost_in_bps,
            fixed_cost=fixed_cost,
            simulator_mode=simulator_mode,
            eval_log_flag = eval_log_flag
        )
        
        sim.cash = initial_cash
        sim.prev_cash = initial_cash

        print(f"\nRunning simulation for interval {start_str} → {end_str}")


        if simulator_mode == "train":

            from dspy.agents.agent_utils import load_training_handler

            #select training function/config based on dqn/other
            train_config_path, train_func = load_training_handler(config)
            train_config = load_config(train_config_path)

            # Create the SimEnv wrapper for training using the simulator
            from dspy.sim.sim_wrapper import SimEnv
            env_fn = lambda: SimEnv(sim)

            # Run the training function with the environment
            train_func(train_config, env_fn=env_fn,run_config= config,features_config=feature_config)
            continue

        else:

            for _ in range(len(df)):
                sim.pre_step()
                sim.step()

            
            # Square off open inventory at the final tick
            last_row = sim.book[sim.ptr - 1]
            sim.square_off(last_row)

            # Save evaluation log
            run_start = interval["start"]
            run_end   = interval["end"]
       
            if eval_log_flag:
                sim.save_eval_log(run_start, run_end)


            print(f"Simulation complete for interval {start_str} → {end_str}")
            print("\n--- Simulation Complete ---")
            print(f"Final Inventory : {sim.inventory}")
            print(f"Final Cash      : {sim.cash}")
            print(f"Final PnL       : {sim.realized_pnl}")

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print('Sim ended at:', now)  


# --- thread params ---
def _configure_torch_threads_cpu():
    import os, torch
    n = int(os.getenv("OMP_NUM_THREADS", "6"))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, n // 2))
    except AttributeError:
        pass
    # optional: print
    print(f"[Torch CPU] threads={torch.get_num_threads()} interop=~{max(1, n//2)}")

# ---------- Entrypoint ----------

if __name__ == "__main__":
    config_path = Path(__file__).parent / "run_config.json"
    config = load_config(config_path)
    run_simulation(config)
