
from pathlib import Path
from typing import Callable
from dspy.agents.dqn.model import load_model_dqn
# from dspy.agents.dqn.model import load_model_pg
from dspy.utils import get_torch_device
import os

# ---------- Load agent dynamically ----------
def get_agent(config: dict, feature_length: int = None) -> object:
    agent_config = config["agent"]
    agent_type = agent_config["type"].lower()
    agent_mode = agent_config["mode"]
    simulator_mode = config["simulator_mode"]

    t_device = get_torch_device(config["device"])

    if agent_type == "dqn":
        from dspy.agents.dqn.agent import DQNAgent
        from dspy.agents.dqn.model import QNetwork

        if agent_mode == "pretrained":
            load_path = Path(__file__).parent/"dqn/saved/run_model/model.pt"
            model=load_model_dqn(load_path,simulator_mode)
        
        else:
            model = QNetwork(feature_length)
        
        agent = DQNAgent(
            model=model,
            tick_size=config["tick_size"],
            min_order_size=config["min_order_size"],
            max_inventory=config["max_inventory"],
            device=t_device,
            
        )
        return agent

    elif agent_type == "pg":
        from dspy.agents.dqn.agent import DQNAgent
        from dspy.agents.dqn.model import QNetwork

        if agent_mode == "pretrained":
            load_path = Path(__file__).parent/"pg/saved/run_model/model.pt"
            model=load_model_pg(load_path,simulator_mode,feature_length)
        
        else:
            model = QNetwork(feature_length)
        
        agent = DQNAgent(
            model=model,
            tick_size=config["tick_size"],
            min_order_size=config["min_order_size"],
            max_inventory=config["max_inventory"],
            
        )
        return agent

    elif agent_type == "symmetric_l1":
        from dspy.agents.symmetric_l1.agent import SymmetricL1Agent
        agent = SymmetricL1Agent(
            tick_size=config["tick_size"],
            min_order_size=config["min_order_size"],
            max_inventory=config["max_inventory"],
        )
        return agent  #  pass agent config



    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def load_training_handler(config: dict) -> tuple[Path, Callable]:
    agent_config = config["agent"]
    agent_type   = agent_config["type"].lower()

    if config["simulator_mode"] != "train":
        raise ValueError("Simulator is not in training mode.")

    # run_dir = Path(__file__).parent.parent.parent / "run"

    if agent_type == "dqn":
        from dspy.agents.dqn.train import train_dqn
        path_training_config = Path(__file__).parent.parent.parent.parent / "run/train_dqn_config.json"
        return path_training_config, train_dqn

    elif agent_type == "pg":
        from dspy.agents.pg.train import train_pg
        path_training_config = Path(__file__).parent.parent.parent.parent / "run/train_pg_config.json"
        return path_training_config, train_pg

    else:
        raise NotImplementedError(f"Training not supported for agent type: {agent_type}")