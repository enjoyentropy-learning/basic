
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def get_gate_data(gate_name, dtype=torch.float):
    """Returns (X, y) tensors for the specified logic gate.

    Args:
        gate_name: 'xor' or 'nand'
        dtype: torch dtype for the tensors
    """
    X = torch.tensor([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ], dtype=dtype)

    truth_tables = {
        'xor':  [0., 1., 1., 0.],
        'nand': [1., 1., 1., 0.],
    }

    if gate_name not in truth_tables:
        raise ValueError(f"Unknown gate '{gate_name}'. Supported: {list(truth_tables.keys())}")

    y = torch.tensor([[v] for v in truth_tables[gate_name]], dtype=dtype)
    return X, y


class RawAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(1, 1, bias=False)
        self.Wk = nn.Linear(1, 1, bias=False)
        self.Wv = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        # x shape: (batch, 2)
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]

        tokens = torch.stack([x1, x2], dim=1)  # (batch, 2, 1)

        Q = self.Wq(tokens)
        K = self.Wk(tokens)
        V = self.Wv(tokens)

        scores = torch.matmul(Q, K.transpose(1, 2))  # (batch, 2, 2)
        attended = torch.matmul(scores, V)           # (batch, 2, 1)

        attended = attended.squeeze(-1)               # (batch, 2)

        return attended


class RawAttentionWithTopNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_attention = RawAttention()
        self.attn_out = nn.Linear(2, 1)  # Bias is required for NAND

    def forward(self, x):
        # x shape: (batch, 2)
        attended = self.raw_attention(x)
        attn_output = self.attn_out(attended + x)  # residual connection
        return attn_output


def initialize_model_and_loss(dtype=torch.float):
    """Initializes the model and loss function."""
    model = RawAttentionWithTopNeuron().to(dtype=dtype)
    loss_fn = nn.MSELoss()
    return model, loss_fn


def _train_adamw_or_sgd(model, optimizer, loss_fn, X, y, epochs, log_interval, log_verbose):
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        loss_history.append(current_loss)

        if log_verbose and epoch % log_interval == 0:
            print(f"Epoch {epoch:4d} | Loss: {current_loss:.8f}")

    return loss_history


def _train_lbfgs(model, optimizer, loss_fn, X, y, epochs, log_interval, log_verbose):
    loss_history = []

    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        current_loss = loss.item()
        loss_history.append(current_loss)

        if log_verbose and epoch % log_interval == 0:
            print(f"Epoch {epoch:4d} | Loss: {current_loss:.8f}")

    return loss_history


def _train_custom_optimizer(model, custom_update_fn, loss_fn, X, y, epochs, log_interval, log_verbose):
    loss_history = []

    for epoch in range(epochs):
        model.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()

        if custom_update_fn:
            custom_update_fn(model, loss)

        current_loss = loss.item()
        loss_history.append(current_loss)

        if log_verbose and epoch % log_interval == 0:
            print(f"Epoch {epoch:4d} | Loss: {current_loss:.8f}")

    return loss_history


def train_model(model, optimizer, loss_fn, X, y,
                epochs=100, optimizer_type='adamw', custom_update_fn=None,
                log_interval=100, log_verbose=True):
    """
    Trains the model and records loss history.

    Args:
        model: The model to train.
        optimizer: The optimizer (None for custom).
        loss_fn: The loss function.
        X: Input tensor (4, 2).
        y: Label tensor (4, 1).
        epochs: Number of training epochs.
        optimizer_type: 'adamw', 'lbfgs', or 'custom'.
        custom_update_fn: Callable with signature (model, loss) for custom updates.
        log_interval: How often to print progress.
        log_verbose: Whether to print logs.

    Returns:
        list: Loss values per epoch.
    """
    model.train()

    if optimizer_type == 'lbfgs':
        return _train_lbfgs(model, optimizer, loss_fn, X, y, epochs, log_interval, log_verbose)
    elif optimizer_type == 'custom':
        return _train_custom_optimizer(model, custom_update_fn, loss_fn, X, y, epochs, log_interval, log_verbose)
    else:
        return _train_adamw_or_sgd(model, optimizer, loss_fn, X, y, epochs, log_interval, log_verbose)


def evaluate_model(model, loss_fn, X, y, gate_name=""):
    """Evaluates the model on all 4 inputs, printing predictions vs targets."""
    model.eval()
    with torch.no_grad():
        preds = model(X)
        loss = loss_fn(preds, y)

        prefix = f"{gate_name} " if gate_name else ""
        print(f"\n{prefix}Evaluation | Loss: {loss.item():.8f}")
        print(f"{'Input':>12}  {'Predicted':>10}  {'Target':>7}")
        for i in range(X.size(0)):
            inp = X[i].tolist()
            pred_val = preds[i].item()
            target_val = y[i].item()
            print(f"  {inp}  {pred_val:10.6f}  {target_val:7.1f}")

    return loss.item()


# Apply newton method on the loss function itself as we know Loss should be zero
# multiply and divide as we can't divide by gradient which is vector
def custom_gradient_update(model, loss):
    """Applies custom gradient update: p -= (loss / ||grad||^2) * grad"""
    with torch.no_grad():
        grad_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm_sq += torch.sum(p.grad ** 2)

        if grad_norm_sq < 1e-12:
            return

        step_scale = loss / grad_norm_sq

        for p in model.parameters():
            if p.grad is not None:
                p -= step_scale * p.grad


dtype_float32 = torch.float
dtype_float64 = torch.float64
