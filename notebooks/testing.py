import json
import sys
from datetime import datetime
from pathlib import Path
import polars as pl


project_root = Path().resolve().parent
sys.path.insert(0, str(project_root / "src"))

from dspy.hdb import get_dataset
from dspy.sim.market_simulator import MarketSimulator
from dspy.utils import to_ns, ts_to_str
from dspy.features.feature_utils import apply_batch_features, extract_features , flatten_features
from dspy.agents.agent_utils import get_agent