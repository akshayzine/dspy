# dspy/agents/dqn/model.py
import math
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    MLP backbone + (optional) Dueling head.
    - Dropout is ACTIVE only in train() (disabled in eval(), so targets are stable).
    - LayerNorm helps with scale drift of LOB features.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int =16,
        hidden_dims=(256, 128),
        dropout_p: float = 0,
        use_layernorm: bool = True,
        dueling: bool = True,
    ):
        super().__init__()
        self.dueling = dueling

        layers = []
        d_in = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d_in, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p and dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            d_in = h
        self.backbone = nn.Sequential(*layers)

        if dueling:
            self.V = nn.Linear(d_in, 1)
            self.A = nn.Linear(d_in, output_dim)
            # zero-init heads for a smooth start
            nn.init.zeros_(self.V.weight); nn.init.zeros_(self.V.bias)
            nn.init.zeros_(self.A.weight); nn.init.zeros_(self.A.bias)
        else:
            self.head = nn.Linear(d_in, output_dim)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            # Kaiming init for ReLU nets
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        if self.dueling:
            A = self.A(z)                    # [B, A]
            V = self.V(z)                    # [B, 1]
            return V + (A - A.mean(dim=1, keepdim=True))
        else:
            return self.head(z)


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
