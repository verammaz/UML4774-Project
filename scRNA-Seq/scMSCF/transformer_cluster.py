# transformer_module.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import sys
import yaml
from tqdm import tqdm
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import data_utils


# -----------------------
# Dataset
# -----------------------
class GeneExpressionDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


# -----------------------
# Transformer Model
# -----------------------
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, d_model=128, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=4*d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x shape: [batch, input_dim]
        x = self.input_linear(x)                  # [batch, d_model]
        x = x.unsqueeze(1)                        # [batch, seq_len=1, d_model]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)                         # global average pooling
        out = self.classifier(x)
        return out


# -----------------------
# Training Utilities
# -----------------------
from tqdm import tqdm

def train_model(model, train_loader, optimizer, criterion, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # update live progress
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} — Average Loss: {avg_loss:.4f}")



def predict_full_dataset(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            logits = model(x)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return preds


# -----------------------
# Main Entry (aligns with scMCF paper)
# -----------------------
def run_transformer_pipeline(
    gene_expression_path,
    confidence_cells_path,
    output_predictions_path,
    meta_path,
    num_heads=4,
    num_layers=2,
    d_model=128,
    dropout=0.1,
    num_epochs=10,
    batch_size=32
):
    print("Loading processed gene expression data...")
    adata = data_utils.load_hd5a(gene_expression_path)

    # Convert to dense if sparse
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    gene_expr = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

    # Load confidence (meta-cluster) CSV
    conf_cells = pd.read_csv(confidence_cells_path)
    conf_cells["cell_name"] = conf_cells["cell_name"].astype(str)

    print(f"Loaded confidence cells: {conf_cells.shape[0]} entries")


    metadata = data_utils.load_metadata(meta_path, label_col='subclass_label')
    n, labels = data_utils.get_sample_labels(metadata, label_col='subclass_label')
    

    # Match common cells by name
    common_cells = gene_expr.index.intersection(conf_cells["cell_name"])
    # common_cells = gene_expr.index.intersection(labels["sample_name"])

    if len(common_cells) == 0:
        raise ValueError("No overlapping cells found between gene expression and confidence cells!")

    print(f"Found {len(common_cells)} overlapping cells")

    # Align matrices
    X_train = gene_expr.loc[common_cells].values
    y_raw = conf_cells.set_index("cell_name").loc[common_cells, "final_cluster"].values 
    #y_raw = labels.set_index("sample_name").loc[common_cells, "subclass_label"].values 

    # Reindex labels to contiguous 0..N-1
    le = LabelEncoder()
    y_train = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    print(f"Detected {num_classes} unique clusters: {list(le.classes_)}")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    input_dim = X_train_scaled.shape[1]
    num_classes = len(np.unique(y_train))

    print(f"Model setup: input_dim={input_dim}, num_classes={num_classes}, "
          f"heads={num_heads}, layers={num_layers}, d_model={d_model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(input_dim, num_classes, num_heads, num_layers, d_model, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Dataset & dataloader
    train_dataset = GeneExpressionDataset(X_train_scaled, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # === Training with tqdm ===
    print(f"Training Transformer for {num_epochs} epochs...")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{epoch_loss/len(train_loader):.4f}"})
        print(f"Epoch {epoch+1} complete (mean loss: {epoch_loss/len(train_loader):.4f})")

    # === Predict full dataset ===
    print("Predicting full dataset...")
    X_full_scaled = scaler.transform(gene_expr.values)
    full_dataset = GeneExpressionDataset(X_full_scaled)
    full_loader = DataLoader(full_dataset, batch_size=batch_size)

    predictions = predict_full_dataset(model, full_loader, device)

    # Save predictions
    predicted_df = pd.DataFrame({
        "cell_name": gene_expr.index,
        "final_cluster": predictions
    })
    predicted_df.to_csv(output_predictions_path, index=False)
    print(f"Saved predicted clusters → {output_predictions_path}")

    return predicted_df


if __name__ == "__main__":
    params_file = sys.argv[1]
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    # Load config
    gene_expr_path = params['expression_processed']
    top_conf_cells_path = params['top_conf_cells_out']
    output_predictions_path = params['output_predictions']
    meta_path = params['metadata']
    num_heads = int(params.get('num_heads', 4))
    num_layers = int(params.get('num_layers', 2))
    d_model = int(params.get('d_model', 128))
    dropout = float(params.get('dropout', 0.1))
    batch_size = int(params.get('batch_size', 32))
    num_epochs = int(params.get('num_epochs', 10))
    lr = float(params.get('learning_rate', 0.001))

    run_transformer_pipeline(gene_expr_path, top_conf_cells_path, output_predictions_path, meta_path,
                             num_heads=num_heads, num_layers=num_layers,
                             num_epochs=num_epochs, d_model=d_model, dropout=dropout)