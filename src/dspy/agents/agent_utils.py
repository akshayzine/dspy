# ---------- Load agent dynamically ----------

def get_agent(config: dict):
    agent_config = config["agent"]
    agent_type = agent_config["type"]
    mode       = agent_config["mode"]
    load_path  = agent_config.get("load_path")

    if agent_type == "DQN":
        from dspy.agents.dqn.agent import DQNAgent
        agent = DQNAgent()
        if mode == "pretrained":
            agent.load(load_path)
        return agent

    elif agent_type == "PG":
        from dspy.agents.pg.agent import PGAgent
        agent = PGAgent()
        if mode == "pretrained":
            agent.load(load_path)
        return agent

    elif agent_type == "symmetric_l2":
        from dspy.agents.symmetric_l2.agent import SymmetricL2Agent
        return SymmetricL2Agent(config)  #  pass agent config



    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
