from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:
    class GATDQN(nn.Module):
        def __init__(self, node_feat_dim: int = 40, hidden_dim: int = 128, num_phases: int = 4):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.value = nn.Linear(hidden_dim, 1)
            self.advantage = nn.Linear(hidden_dim, num_phases)

        def forward(self, x):
            h = self.encoder(x)
            value = self.value(h)
            adv = self.advantage(h)
            return value + adv - adv.mean(dim=-1, keepdim=True)
else:
    class GATDQN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for GATDQN")
