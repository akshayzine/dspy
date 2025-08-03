import json
import sys
from datetime import datetime
from pathlib import Path
import polars as pl


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dspy.hdb import get_dataset
from dspy.sim.market_simulator import MarketSimulator
from dspy.utils import to_ns, ts_to_str
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
    max_inventory    = config["max_inventory"]
    inv_penalty      = config["inventory_penalty"]
    initial_cash     = config["initial_cash"]
    agent_config     = config["agent"]
    intervals        = config["intervals"]
    min_order_size   = config["min_order_size"]
    tick_size        = config["tick_size"]
    initial_cash     = config["initial_cash"]
    cost_in_bps      = config["cost_in_bps"]
    fixed_cost       = config["fixed_cost"]
    simulator_mode   = config["simulator_mode"]

    loader = get_dataset(dataset_name)
    all_books, all_ts = [], []

    # ---------- Load cofeature config file ----------
    feature_path = Path(__file__).parent / "features.json"
    feature_config = load_config(feature_path)
    inventory_feature_flag = "inventory" in feature_config.keys()

    for interval in intervals:
        start_str = interval["start"]
        end_str   = interval["end"]

        start_ts = datetime.strptime(interval["start"], "%Y-%m-%d %H:%M:%S").strftime("%y%m%d.%H%M%S")
        end_ts   = datetime.strptime(interval["end"],   "%Y-%m-%d %H:%M:%S").strftime("%y%m%d.%H%M%S")

        df = loader.load_book(
            product=product,
            times=[start_ts, end_ts],
            depth=depth,
            type="book_snapshot_25",
            lazy=False
        )

        df, feature_cols = apply_batch_features(df, feature_config)

        agent = get_agent(config)

        sim = MarketSimulator(
            book=df,
            feature_config=feature_config,
            inventory_feature_flag = inventory_feature_flag,
            feature_columns=feature_cols,
            agent=agent,
            latency_ns=latency_ns,
            inventory_penalty=inv_penalty,
            max_inventory=max_inventory,
            min_order_size=min_order_size,
            tick_size=tick_size,
            initial_cash=initial_cash,
            cost_in_bps=cost_in_bps,
            fixed_cost=fixed_cost,
            simulator_mode=simulator_mode
        )
        
        sim.cash = initial_cash
        sim.prev_cash = initial_cash

        print(f"\nRunning simulation for interval {start_str} → {end_str}")
        for _ in range(11):
            sim.step()
            print('sim_inventory:', sim.inventory)

        
        print('sim_inventory:', sim.inventory)
        # Square off open inventory at the final tick
        last_row = sim.book[sim.ptr - 1]
        sim.square_off(last_row)

        # Save evaluation log
        run_start = interval["start"]
        run_end   = interval["end"]
        print(run_start)
        sim.save_eval_log(run_start, run_end)


        print(f"Simulation complete for interval {start_str} → {end_str}")
        print("\n--- Simulation Complete ---")
        print(f"Final Inventory : {sim.inventory}")
        print(f"Final Cash      : {sim.cash}")
        print(f"Final PnL       : {sim.realized_pnl}")


# ---------- Entrypoint ----------

if __name__ == "__main__":
    config_path = Path(__file__).parent / "run_config.json"
    config = load_config(config_path)
    run_simulation(config)
