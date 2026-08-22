import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold 

# ============================================================
# Meta stacking with an Optimized PyTorch Tabular MLP
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class TabularMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Switched to LayerNorm since input features are concatenated probabilities [0, 1]
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.30),  # Slightly higher dropout to prevent meta-overfitting
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)

model_names = sorted(preds.keys())
if not model_names:
    raise RuntimeError("No OOF predictions were loaded; cannot build a meta stack.")

if not all(name in prob_data for name in model_names):
    missing = [name for name in model_names if name not in prob_data]
    raise RuntimeError(f"Missing test probabilities for: {missing}")


def probability_to_logits(prob_array, epsilon=1e-7):
    """
    Stabilizes and transforms probabilities into log-odds (logits).
    Works for both binary and multiclass probability matrices.
    """
    # Clip to avoid log(0) or division by zero errors
    prob_clipped = np.clip(prob_array, epsilon, 1.0 - epsilon)
    return np.log(prob_clipped / (1.0 - prob_clipped))

# Concatenate and transform OOF predictions
X_train_list = []
for name in model_names:
    raw_probs = preds[name].astype(np.float32)
    logit_features = probability_to_logits(raw_probs)
    X_train_list.append(logit_features)

X_train = np.concatenate(X_train_list, axis=1)

# Concatenate and transform Test predictions
X_test_list = []
for name in model_names:
    raw_test_probs = prob_data[name][prob_cols].to_numpy(dtype=np.float32)
    logit_test_features = probability_to_logits(raw_test_probs)
    X_test_list.append(logit_test_features)

X_test = np.concatenate(X_test_list, axis=1)


# Train a 5-fold CV meta-model.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_meta = np.zeros((len(y), 3), dtype=np.float32)

# Array to collect test predictions from each fold model
test_preds_folds = np.zeros((len(X_test), 3), dtype=np.float32)

EPOCHS = 35  # Increased epochs to give the network time to converge
BATCH_SIZE = 1024  # Reduced batch size for better gradient granularity
VERBOSE = 2

X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y)):
    print(f"\n--- Training Fold {fold + 1} ---")
    X_tr = torch.tensor(X_train[tr_idx], dtype=torch.float32, device=DEVICE)
    y_tr = torch.tensor(y[tr_idx], dtype=torch.long, device=DEVICE)
    X_va = torch.tensor(X_train[va_idx], dtype=torch.float32, device=DEVICE)
    y_va = torch.tensor(y[va_idx], dtype=torch.long, device=DEVICE)

    counts = torch.bincount(y_tr)
    class_weights = counts.sum() / (len(counts) * counts.float())
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    model = TabularMLP(input_dim=X_tr.shape[1], num_classes=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    train_ds = TensorDataset(X_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_score = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_dl:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va)
            val_pred = torch.argmax(val_logits, dim=1)
            score = balanced_accuracy_score(y_va.cpu().numpy(), val_pred.cpu().numpy())

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        
        if epoch % VERBOSE == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:4d} | Balanced Accuracy: {score:.5f}")

    print(f"Fold {fold + 1} Best Balanced Accuracy: {best_score:.5f}")
    
    # Load best checkpoint for OOF and Test Inference
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        oof_meta[va_idx] = model(X_va).softmax(dim=1).cpu().numpy()
        # Accumulate test predictions (Blends predictions across all fold models)
        test_preds_folds += model(X_test_t).softmax(dim=1).cpu().numpy() / cv.n_splits

# Final CV score tracking
meta_score = balanced_accuracy_score(y, np.argmax(oof_meta, axis=1))
print(f"\nMeta stacking CV balanced accuracy: {meta_score:.6f}")

# Output Processing
output_dir = Path("solution_metadata/meta_stack")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "meta_mlp_stack_v6.csv"

blend = pd.DataFrame({"id": prob_data[model_names[0]]["id"]})
blend[prob_cols] = test_preds_folds
blend["class"] = blend[prob_cols].idxmax(axis=1)
blend[["id", "class"]].to_csv(output_path, index=False)

print("\n==================================================")
print("Meta Stacking Submission Generated")
print("==================================================")
print(f"CV balanced accuracy : {meta_score:.6f}")
print(f"Saved submission     : {output_path}")
print(f"Base models used     : {len(model_names)}")