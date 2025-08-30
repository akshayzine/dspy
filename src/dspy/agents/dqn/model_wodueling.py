# dqn/model.py
import torch.nn as nn
import torch

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim = 16):
        super().__init__()
        layer_1 = 256
        layer_2 = 128
        self.net = nn.Sequential(
            
            nn.Linear(input_dim, layer_1),
            nn.LayerNorm(layer_1),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1),
            nn.Linear(layer_1, layer_2),
            nn.LayerNorm(layer_2),
            nn.ReLU(inplace=True),
            nn.Linear(layer_2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def load_model(path: str, expected_input_dim: int = None, device: str = "cpu") -> QNetwork:
    state_dict = torch.load(path, map_location=device)

    # Extract input/output dims from state_dict
    weight_keys = [k for k in state_dict if "weight" in k]
    input_dim = state_dict[weight_keys[0]].shape[1]
    output_dim = state_dict[weight_keys[-1]].shape[0]

    # Optional sanity check on input_dim
    if expected_input_dim is not None and input_dim != expected_input_dim:
        raise ValueError(f"Feature length mismatch: model expects input dim {input_dim}, but got {expected_input_dim}")

    # Infer model architecture from saved weights
    model = QNetwork(input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
