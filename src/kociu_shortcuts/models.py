"""Small paired multilayer perceptrons used by the confirmatory pipeline."""

from __future__ import annotations
import random
import numpy as np

def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for training. Install the 'train' dependencies first.") from exc
    return torch, nn

def set_all_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch, _ = require_torch(); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def build_mlp(input_dim: int = 4, hidden_dim: int = 16, depth: int = 2, activation: str = "tanh"):
    torch, nn = require_torch()
    activations = {"tanh": nn.Tanh, "relu": nn.ReLU}
    if activation not in activations:
        raise ValueError(f"unsupported activation: {activation}")
    if depth < 1:
        raise ValueError("depth must be at least one")
    activation_cls = activations[activation]
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [nn.Linear(input_dim, hidden_dim), activation_cls()]
            for _ in range(depth - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), activation_cls()])
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_dim, 1)
        def hidden_rep(self, x): return self.body(x)
        def forward(self, x): return torch.sigmoid(self.head(self.hidden_rep(x)))
    return MLP()

def train_full_batch(model, X: np.ndarray, y: np.ndarray, *, epochs: int = 1000, learning_rate: float = 0.01, checkpoints: list[int] | None = None):
    torch, nn = require_torch()
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss(); checkpoint_set = set(checkpoints or [epochs])
    if any(epoch < 1 or epoch > epochs for epoch in checkpoint_set):
        raise ValueError("training checkpoints must lie between 1 and epochs")
    history = []; model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(); loss = loss_fn(model(X_t), y_t); loss.backward(); optimizer.step()
        if epoch in checkpoint_set:
            history.append({"epoch": epoch, "training_loss": float(loss.detach().cpu())})
    return history

def hidden_numpy(model, X: np.ndarray) -> np.ndarray:
    torch, _ = require_torch(); model.eval()
    with torch.no_grad():
        return model.hidden_rep(torch.as_tensor(X, dtype=torch.float32)).cpu().numpy()

def predict_numpy(model, X: np.ndarray) -> np.ndarray:
    torch, _ = require_torch(); model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).cpu().numpy().reshape(-1)
