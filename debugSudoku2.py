
import itertools
import random
import torch
import torch.nn as nn
import torch.optim as optim

def generate_data(num_symbols=9, train_frac=0.003, seed=0, dtype=torch.float):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)

    # -----------------------------
    # Generate ALL valid permutations
    # -----------------------------
    valid_sequences = list(itertools.permutations(range(1, num_symbols + 1)))
    valid_labels = [1.0] * len(valid_sequences)

    # -----------------------------
    # Generate invalid sequences
    # -----------------------------
    def generate_invalid(num_samples):
        invalid = set()
        while len(invalid) < num_samples:
            seq = [random.randint(1, num_symbols) for _ in range(num_symbols)]
            if len(set(seq)) != num_symbols:  # invalid condition
                invalid.add(tuple(seq))
        return list(invalid)

    num_invalid = len(valid_sequences) # Ensure equal number of valid and invalid for balance
    invalid_sequences = generate_invalid(num_invalid)
    invalid_labels = [0.0] * num_invalid

    # -----------------------------
    # Combine & shuffle
    # -----------------------------
    all_sequences = valid_sequences + invalid_sequences
    all_labels = valid_labels + invalid_labels

    combined = list(zip(all_sequences, all_labels))
    random.shuffle(combined)

    # -----------------------------
    # Train / validation split
    # -----------------------------
    split_idx = int(len(combined) * train_frac)

    train_data = combined[:split_idx]
    val_data   = combined[split_idx:]

    # -----------------------------
    # Ensure NO overlap (sanity check)
    # -----------------------------
    train_set = set(seq for seq, _ in train_data)
    val_set   = set(seq for seq, _ in val_data)

    assert train_set.isdisjoint(val_set), "Train / validation overlap detected!"

    # -----------------------------
    # Convert to tensors
    # -----------------------------
    def to_tensors(data):
        X = torch.tensor([seq for seq, _ in data], dtype=torch.long)
        y = torch.tensor([[label] for _, label in data], dtype=dtype)
        return X, y

    X_train_original, y_train = to_tensors(train_data)
    X_val_original,   y_val   = to_tensors(val_data)

    # -----------------------------
    # One-hot encode
    # -----------------------------
    X_train_oh = torch.nn.functional.one_hot(X_train_original - 1, num_classes=num_symbols).to(dtype=dtype)
    X_val_oh   = torch.nn.functional.one_hot(X_val_original - 1,   num_classes=num_symbols).to(dtype=dtype)

    # -----------------------------
    # Final shapes
    # -----------------------------
    print("Train OH:", X_train_oh.shape, "Train y:", y_train.shape)
    print("Val OH:  ", X_val_oh.shape, "Val y:", y_val.shape)

    return X_train_oh, y_train, X_val_oh, y_val, X_train_original, X_val_original



class MultiplyBlock(nn.Module):
    def __init__(self, d_model, dtype=torch.float):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.Wk = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.Wv = nn.Linear(d_model, d_model, bias=False, dtype=dtype)

    def forward(self, x):
        # x: (batch, seq_len, d_model)

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = torch.matmul(Q, K.transpose(1, 2))  # (batch, seq_len, seq_len)
        out = torch.matmul(scores, V)                # (batch, seq_len, d_model)

        return out
    
class PolynomialAttentionNet(nn.Module):
    def __init__(self, input_dim, d_model, depth, dtype=torch.float):
        super().__init__()

        # Per-token linear embedding
        self.embed = nn.Linear(input_dim, d_model, bias=True, dtype=dtype)

        # Polynomial expansion layers
        self.layers = nn.ModuleList([
            MultiplyBlock(d_model, dtype=dtype)
            for _ in range(depth)
        ])

        # Final aggregation
        self.output = nn.Linear(d_model, 1, bias=True, dtype=dtype)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)

        h = self.embed(x)   # (batch, seq_len, d_model)

        for layer in self.layers:
            h = h + 0.1 * layer(h)   # scaled residual for stability

        # Aggregate across tokens
        h = h.sum(dim=1)   # (batch, d_model)

        return self.output(h)


class PLTBlock(nn.Module):
    """
    Polynomial Logic Transformer Block
    Uses raw multiplication for logic gates. No ReLU, no Softmax.
    """
    def __init__(self, d_model,  dtype=torch.float):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.Wk = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.Wv = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        
        # The FFN here is just a Linear projection to 
        # redistribute the 'logic signals' without squashing.
        self.proj = nn.Linear(d_model, d_model, bias=True, dtype=dtype)

    def forward(self, x):
        # 1. Generate Product Terms (AND gates)
        # (batch, seq, d_model) @ (batch, d_model, seq) -> (batch, seq, seq)
        attn = torch.matmul(self.Wq(x), self.Wk(x).transpose(-1, -2))
        
        # 2. Aggregate Product Terms (XOR/Summation gates)
        # (batch, seq, seq) @ (batch, seq, d_model) -> (batch, seq, d_model)
        z = torch.matmul(attn, self.Wv(x))
        
        # 3. Residual connection allows lower-degree polynomials 
        # to pass through to the next layer.
        return self.proj(z) + x

class SymbolicLogicEngine(nn.Module):
    def __init__(self, vocab_size, n_layers=3,  dtype=torch.float):
        super().__init__()
        self.d_model = vocab_size
        self.layers = nn.ModuleList([PLTBlock(self.d_model, dtype=dtype) for _ in range(n_layers)])
        
        # Final Readout: A single polynomial sum to determine the truth value
        self.readout = nn.Linear(self.d_model, 1, dtype=dtype)

    def forward(self, x):
        # x is one-hot: (batch, seq_len, vocab_size)
        for layer in self.layers:
            x = layer(x)
            
        # For sequence-level logic (like uniqueness), we sum across positions
        # This is equivalent to an 'OR' gate across the whole sequence.
        logical_sum = x.mean(dim=1)   # (batch, vocab_size) the output shape
        return self.readout(logical_sum)


# 3. Create a function named initialize_model_and_loss
def initialize_model_and_loss(num_symbols, d_model, depth,  dtype=torch.float):

    """Initializes the model and loss function."""
    #model = PolynomialAttentionNet(num_symbols, d_model, depth, dtype)
    model = SymbolicLogicEngine(d_model, depth, dtype)
    loss_fn = nn.MSELoss()
    return model, loss_fn



def _train_adamw_or_sgd(model, optimizer, loss_fn, X_train_oh, y_train, epochs, batch_size, log_interval, log_verbose):
    loss_history = []
    N = X_train_oh.size(0)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0.0
        epoch_total = 0.0

        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            X_batch = X_train_oh[idx]
            y_batch = y_train[idx]

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds_bin = (preds > 0.5).float()
                epoch_loss += loss.item() * y_batch.size(0)
                epoch_correct += (preds_bin == y_batch).sum().item()
                epoch_total += y_batch.size(0)

        current_loss = epoch_loss / epoch_total if epoch_total > 0 else 0.0
        current_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        loss_history.append(current_loss)

        if log_verbose and epoch % log_interval == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {current_loss:.8f} | "
                f"Acc: {current_acc:.8f}"
            )
    return loss_history

def _train_lbfgs(model, optimizer, loss_fn, X_train_oh, y_train, epochs, log_interval, log_verbose):
    loss_history = []
    N = X_train_oh.size(0)

    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            preds = model(X_train_oh)
            loss = loss_fn(preds, y_train)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        current_loss = loss.item()

        with torch.no_grad():
            preds = model(X_train_oh)
            preds_bin = (preds > 0.5).float()
            current_acc = (preds_bin == y_train).float().mean().item()

        loss_history.append(current_loss)

        if log_verbose and epoch % log_interval == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {current_loss:.8f} | "
                f"Acc: {current_acc:.8f}"
            )
    return loss_history

def _train_custom_optimizer(model, custom_update_fn, loss_fn, X_train_oh, y_train, epochs, log_interval, log_verbose):
    loss_history = []
    N = X_train_oh.size(0)

    for epoch in range(epochs):
        model.zero_grad()
        preds = model(X_train_oh)
        loss = loss_fn(preds, y_train)
        loss.backward()
        if custom_update_fn:
            custom_update_fn(model, loss)
        current_loss = loss.item()

        with torch.no_grad():
            preds_bin = (preds > 0.5).float()
            current_acc = (preds_bin == y_train).float().mean().item()

        loss_history.append(current_loss)

        if log_verbose and epoch % log_interval == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {current_loss:.8f} | "
                f"Acc: {current_acc:.8f}"
            )
    return loss_history

def train_model(model, optimizer, loss_fn, X_train_oh, y_train, X_val_oh=None, y_val=None,
                epochs=100, batch_size=None, optimizer_type='adamw', custom_update_fn=None,
                log_interval=100, log_verbose=True):
    """
    Trains the model and records loss history.

    Args:
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        loss_fn (nn.Module): The loss function.
        X_train_oh (torch.Tensor): One-hot encoded training features.
        y_train (torch.Tensor): Training labels.
        X_val_oh (torch.Tensor, optional): One-hot encoded validation features.
        y_val (torch.Tensor, optional): Validation labels.
        epochs (int): Number of training epochs/steps.
        batch_size (int, optional): Batch size for training. If None, uses full batch.
        optimizer_type (str): 'adamw', 'lbfgs', or 'custom'.
        custom_update_fn (callable, optional): A function for custom parameter updates.
                                               Expected signature: custom_update_fn(model, loss)
        log_interval (int): How often to print training progress.
        log_verbose (bool): Whether to print detailed logs.

    Returns:
        list: List of training loss values per epoch/step.
    """
    model.train()
    N = X_train_oh.size(0)
    batch_size = batch_size if batch_size is not None else N

    if optimizer_type == 'lbfgs':
        return _train_lbfgs(model, optimizer, loss_fn, X_train_oh, y_train, epochs, log_interval, log_verbose)
    elif optimizer_type == 'custom':
        return _train_custom_optimizer(model, custom_update_fn, loss_fn, X_train_oh, y_train, epochs, log_interval, log_verbose)
    else:  # 'adamw', 'sgd', or any other batch-based optimizer
        return _train_adamw_or_sgd(model, optimizer, loss_fn, X_train_oh, y_train, epochs, batch_size, log_interval, log_verbose)



def evaluate_model(model, loss_fn, X_data_oh, y_data, X_data_original=None, data_name="Validation"):
    """
    Evaluates the model's performance on a given dataset.

    Args:
        model (nn.Module): The trained model.
        loss_fn (nn.Module): The loss function.
        X_data_oh (torch.Tensor): One-hot encoded features of the dataset.
        y_data (torch.Tensor): Labels of the dataset.
        X_data_original (torch.Tensor, optional): Original (non-one-hot) features for logging misclassified examples.
        data_name (str): Name of the dataset (e.g., "Training", "Validation").

    Returns:
        tuple: (loss, accuracy)
    """
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient tracking
        preds = model(X_data_oh) # Make predictions
        loss = loss_fn(preds, y_data)
        preds_bin = (preds > 0.5).float() # Binarize predictions

        correct = (preds_bin == y_data).sum().item()
        total = y_data.size(0)
        accuracy = correct / total

        print(
            f"{data_name} results | "
            f"Loss: {loss.item():.4f} | "
            f"Acc: {accuracy:.4f}"
        )

        if X_data_original is not None:
            incorrect_indices = (preds_bin != y_data).nonzero(as_tuple=True)[0]
            if len(incorrect_indices) > 0:
                print(f"\nAnalyzing misclassified {data_name} examples ({len(incorrect_indices)} total):")
                for i in incorrect_indices[:5]: # Show up to 5 misclassified examples
                    original_input = X_data_original[i].tolist()
                    predicted_value = preds[i].item()
                    true_label = y_data[i].item()
                    print(f"  Input: {original_input}, Predicted: {predicted_value:.4f}, Actual: {int(true_label)}")
            else:
                print(f"\nAll {data_name} examples predicted correctly!")

    return loss.item(), accuracy


import torch.optim as optim
import matplotlib.pyplot as plt

# Custom update rule function (from previous implementation 'J4vlvcQ1QWKi')
def old_custom_gradient_update(model, loss):
    """Applies a custom gradient update rule."""
    with torch.no_grad():
        grad_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm_sq += torch.sum(p.grad ** 2)

        if grad_norm_sq < 1e-12:
            # print("Gradient vanished — skipping update.") # Commented out to avoid excessive output
            return

        step_scale = loss / grad_norm_sq

        for p in model.parameters():
            if p.grad is not None:
                p -= step_scale * p.grad

# Apply newton method on the loss function itself as we know Loss should be zero
# multiply and divide as we can't divide by gradient which is vector
def custom_gradient_update(model, loss):
    """Applies custom gradient update: p -= (loss / ||grad||^2) * grad"""
    with torch.no_grad():
        eps = torch.finfo(loss.dtype).eps
        threshold = eps ** 2

        grad_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm_sq += torch.sum(p.grad ** 2)

        if grad_norm_sq < threshold:
            print(
                f"Gradient vanished | "
                f"grad_norm_sq={grad_norm_sq.item():.3e}, "
                f"threshold={threshold:.3e}, "
                f"dtype={loss.dtype}"
            )
            return

        step_scale = loss / grad_norm_sq

        for p in model.parameters():
            if p.grad is not None:
                p -= step_scale * p.grad

# 2. Initialize d_model and seq_len
num_symbols = 9
d_model = 32
seq_len = 9
depth = 2

# Define the dtypes to experiment with
dtype_float32 = torch.float
dtype_float64 = torch.float64


##print("--- Training with AdamW Optimizer (float32) ---")
# # torch.set_default_dtype(dtype_float32) # Set default dtype for this block
# # # 3. Generate training and validation data for AdamW
# # X_train_oh_adamw, y_train_adamw, X_val_oh_adamw, y_val_adamw, X_train_original_adamw, X_val_original_adamw = generate_data(num_symbols=9, train_frac=0.003, seed=42, dtype=dtype_float32)

# # # 4. Initialize a new model and loss function
# # model_adamw, loss_fn_adamw = initialize_model_and_loss(d_model, seq_len, dtype=dtype_float32)

# # 5. Initialize an AdamW optimizer
# optimizer_adamw = optim.AdamW(model_adamw.parameters(), lr=0.03)

# # 6. Train the model using train_model with AdamW optimizer
# adamw_loss_history = train_model(model_adamw, optimizer_adamw, loss_fn_adamw, X_train_oh_adamw, y_train_adamw,
#                                  epochs=500, log_interval=100, optimizer_type='adamw')

# # 7. Evaluate the AdamW-trained model on the validation set
# print("\n--- Evaluation for AdamW Model (float32) ---")
# evaluate_model(model_adamw, loss_fn_adamw, X_val_oh_adamw, y_val_adamw, X_val_original_adamw, data_name="Validation (AdamW, float32)")


# print("\n--- Training with LBFGS Optimizer (float64) ---")
# torch.set_default_dtype(dtype_float64) # Set default dtype for this block
# # 8. Generate new training and validation data for LBFGS
# X_train_oh_lbfgs, y_train_lbfgs, X_val_oh_lbfgs, y_val_lbfgs, X_train_original_lbfgs, X_val_original_lbfgs = generate_data(num_symbols=9, train_frac=0.003, seed=43, dtype=dtype_float64)

# # 9. Initialize another new model and loss function
# model_lbfgs, loss_fn_lbfgs = initialize_model_and_loss(d_model, seq_len, dtype=dtype_float64)

# # 10. Initialize an LBFGS optimizer
# optimizer_lbfgs = torch.optim.LBFGS(model_lbfgs.parameters(), lr=1.0, max_iter=20, history_size=100, line_search_fn="strong_wolfe")

# # 11. Train the model using train_model with LBFGS optimizer
# lbfgs_loss_history = train_model(model_lbfgs, optimizer_lbfgs, loss_fn_lbfgs, X_train_oh_lbfgs, y_train_lbfgs,
#                                  epochs=400, optimizer_type='lbfgs', log_interval=100) # LBFGS steps are not epochs, usually fewer steps

# # 12. Evaluate the LBFGS-trained model on the validation set
# print("\n--- Evaluation for LBFGS Model (float64) ---")
# evaluate_model(model_lbfgs, loss_fn_lbfgs, X_val_oh_lbfgs, y_val_lbfgs, X_val_original_lbfgs, data_name="Validation (LBFGS, float64)")


# print("\n--- Training with Custom Optimizer (Gradient Normalization, float32) ---")
# torch.set_default_dtype(dtype_float32) # Set default dtype for this block
# # 13. Generate new training and validation data for Custom Optimizer
# X_train_oh_custom, y_train_custom, X_val_oh_custom, y_val_custom, X_train_original_custom, X_val_original_custom = generate_data(num_symbols=9, train_frac=0.003, seed=44, dtype=dtype_float32)

# # 14. Initialize yet another new model and loss function
# model_custom, loss_fn_custom = initialize_model_and_loss(d_model, seq_len, dtype=dtype_float32)

# # 15. Train the model using train_model with custom update rule
# custom_loss_history = train_model(model_custom, None, loss_fn_custom, X_train_oh_custom, y_train_custom,
#                                   epochs=500, optimizer_type='custom', custom_update_fn=custom_gradient_update, log_interval=100)

# # 16. Evaluate the custom-optimizer-trained model on the validation set
# print("\n--- Evaluation for Custom Optimizer Model (float32) ---")
# evaluate_model(model_custom, loss_fn_custom, X_val_oh_custom, y_val_custom, X_val_original_custom, data_name="Validation (Custom, float32)")



# print("\n--- Training with Different Data Parameters (AdamW, 5 symbols, float32) ---")
# torch.set_default_dtype(dtype_float32) # Set default dtype for this block
# d_model_small = 5
# seq_len_small = 5
# X_train_oh_small, y_train_small, X_val_oh_small, y_val_small, X_train_original_small, X_val_original_small = generate_data(num_symbols=d_model_small, train_frac=0.1, seed=45, dtype=dtype_float32)
# model_small, loss_fn_small = initialize_model_and_loss(d_model_small, seq_len_small, dtype=dtype_float32)
# optimizer_small = optim.AdamW(model_small.parameters(), lr=0.01)
# small_data_loss_history = train_model(model_small, optimizer_small, loss_fn_small, X_train_oh_small, y_train_small,
#                                       epochs=300, log_interval=50, optimizer_type='adamw')
# evaluate_model(model_small, loss_fn_small, X_val_oh_small, y_val_small, X_val_original_small, data_name="Validation (Small Data, float32)")


# # --- Visualize Training History ---
# plt.figure(figsize=(12, 8))
# plt.plot(adamw_loss_history, label='AdamW Loss (float32)', alpha=0.7)
# plt.plot(lbfgs_loss_history, label='LBFGS Loss (float64)', alpha=0.7)
# plt.plot(custom_loss_history, label='Custom Update Loss (float32)', alpha=0.7)
# plt.plot(small_data_loss_history, label='AdamW (5 Symbols, float32) Loss', alpha=0.7, linestyle='--')
# plt.xlabel('Epoch/Step')
# plt.ylabel('Loss')
# plt.title('Training Loss History for Different Optimizers and Data Parameters')
# plt.legend()
# plt.grid(True)
# plt.yscale('log') # Use log scale for better visualization of different convergence speeds
# plt.show()

# # --- Final Task: Summary ---
# print("\n--- Summary of Organized Code Structure ---")
# print("The code has been refactored into modular functions:")
# print("1. `generate_data`: Centralized function to create training and validation datasets, ensuring reproducibility and easy modification of data parameters (e.g., number of symbols, train/validation split ratio).")
# print("2. `initialize_model_and_loss`: A function to instantiate the `RawAttentionWithTopNeuron` model and the `MSELoss` function, making it easy to configure model size or loss types.")
# print("3. `train_model`: A flexible training loop that accommodates different optimizers (AdamW, LBFGS, custom update rules) by handling their specific `step` mechanisms (e.g., LBFGS's `closure`). This allows for straightforward experimentation with various optimization algorithms.")
# print("4. `evaluate_model`: A dedicated function to assess model performance on given data, calculating loss and accuracy, and providing insights into misclassified examples. This standardizes the evaluation process.")

# print("\n**How this structure facilitates experimentation:**")
# print("- **Different Optimizers**: By passing different optimizer instances (e.g., `optim.AdamW`, `torch.optim.LBFGS`) and specifying `optimizer_type`, the same `train_model` function can be used without code duplication, as demonstrated above.")
# print("- **Different Input Data Parameters**: The `generate_data` function allows easy generation of datasets with varying `num_symbols` or `train_frac`. This means one can quickly test the model's performance on different problem complexities or dataset sizes, as shown with the 5-symbol example.")
# print("- **Configurable Data Types**: The added `dtype` parameter to `generate_data` and `initialize_model_and_loss` functions allows for easy experimentation with different floating-point precision (e.g., `torch.float32` vs `torch.float64`) for both data and model parameters, enabling analysis of its impact on training stability and performance.")
# print("- **Clearer Workflow**: The modularity makes the entire machine learning pipeline (data -> model -> train -> evaluate) much clearer, easier to understand, debug, and extend.")
# print("- **Reproducibility**: Explicit seeding and function encapsulation promote reproducibility of experiments.")

