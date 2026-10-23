import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, 
                             confusion_matrix, classification_report, log_loss)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from tqdm import tqdm

# ==========================================
# 1. Configuration & Setup
# ==========================================
MAIN_FL_ROUNDS = 25 
LOCAL_EPOCHS = 10     
BATCH_SIZE = 32
PATIENCE = 5

TARGET_CLASSES = ['A', 'B', 'D', 'E'] 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(f"Executing on device: {DEVICE}") 

os.makedirs('visualizations', exist_ok=True) 
os.makedirs('reports', exist_ok=True) 
os.makedirs('models', exist_ok=True) 

# Hardcoded best parameters (MOON typically uses mu=1 or mu=5, and tau=0.5)
best_params = {
    'lr': 0.004160308766542998, 
    'mu': 1.0, # MOON Contrastive Loss Weight (lambda in some texts, mu in paper)
    'tau': 0.5, # Temperature for contrastive loss
    'filter1': 64, 
    'filter2': 128, 
    'dropout_rate': 0.3451319786446383, 
    'kernel_size': 7
}

# ==========================================
# 2. Strict Train-Only Preprocessing
# ==========================================
print("Loading gravity tilt datasets...") 
train_df = pd.read_csv('tilt_fl_train.csv') 
val_df = pd.read_csv('tilt_fl_validation.csv') 
test_df = pd.read_csv('tilt_fl_test.csv') 

print(f"Filtering dataset down to 4 Target Classes: {TARGET_CLASSES}...") 
train_df = train_df[train_df['label'].isin(TARGET_CLASSES)] 
val_df = val_df[val_df['label'].isin(TARGET_CLASSES)] 
test_df = test_df[test_df['label'].isin(TARGET_CLASSES)] 

base_features = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'pitch', 'roll'] 
feature_cols = [] 
for feature in base_features:
    feature_cols.extend([f'{feature}_{i}' for i in range(1, 61)]) 

scaler = StandardScaler() 
train_features = scaler.fit_transform(train_df[feature_cols]) 
val_features = scaler.transform(val_df[feature_cols]) 
test_features = scaler.transform(test_df[feature_cols]) 

le = LabelEncoder() 
train_labels = le.fit_transform(train_df['label']) 
val_labels = le.transform(val_df['label']) 
test_labels = le.transform(test_df['label']) 

train_df = train_df.copy() 
val_df = val_df.copy() 
test_df = test_df.copy() 

train_df[feature_cols] = train_features 
val_df[feature_cols] = val_features 
test_df[feature_cols] = test_features 

train_df['label'] = train_labels 
val_df['label'] = val_labels 
test_df['label'] = test_labels 
num_classes = len(le.classes_) 
class_names = list(le.classes_) 

def get_tensor_dataset(df):
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32) 
    y = torch.tensor(df['label'].values, dtype=torch.long) 
    return TensorDataset(X, y) 

full_train_loader = DataLoader(get_tensor_dataset(train_df), batch_size=BATCH_SIZE, shuffle=True) 
val_loader = DataLoader(get_tensor_dataset(val_df), batch_size=BATCH_SIZE, shuffle=False) 
test_loader = DataLoader(get_tensor_dataset(test_df), batch_size=BATCH_SIZE, shuffle=False) 

client_ids = list(train_df['client_id'].unique()) 
client_loaders = {} 

for cid in client_ids:
    client_data = train_df[train_df['client_id'] == cid] 
    if len(client_data) > 0:
        client_loaders[cid] = DataLoader(get_tensor_dataset(client_data), batch_size=BATCH_SIZE, shuffle=True) 

active_client_ids = list(client_loaders.keys()) 
top_3_clients = train_df['client_id'].value_counts().nlargest(3).index.tolist() 

print(f"Total Active Clients: {len(active_client_ids)} (100% Participation)") 

# ==========================================
# 3. Weighted Cross-Entropy Loss & FedAvg
# ==========================================
weights_list = [1.0, 1.0, 1.5, 1.5] if num_classes == 4 else [1.0] * num_classes 
class_weights = torch.tensor(weights_list, dtype=torch.float32).to(DEVICE) 
criterion = nn.CrossEntropyLoss(weight=class_weights) 

def weighted_fedavg(local_states, client_sizes, reference_state):
    total_samples = float(sum(client_sizes)) 
    aggregated_state = {} 
    for key in reference_state.keys():
        weighted_sum = torch.zeros_like(reference_state[key], dtype=torch.float32) 
        for state, n_samples in zip(local_states, client_sizes):
            weighted_sum += (n_samples / total_samples) * state[key].float() 
        aggregated_state[key] = weighted_sum 
    return aggregated_state 

# ==========================================
# 4. Model Architecture 
# ==========================================
class OptimizedSensorCNN(nn.Module): 
    def __init__(self, num_classes, filter1, filter2, dropout_rate, kernel_size):
        super().__init__() 
        padding = kernel_size // 2  
        
        self.conv1 = nn.Conv1d(8, filter1, kernel_size=kernel_size, padding=padding) 
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=filter1) 
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) 
        
        self.conv2 = nn.Conv1d(filter1, filter2, kernel_size=kernel_size, padding=padding) 
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=filter2) 
        
        with torch.no_grad():
            dummy_x = torch.zeros(1, 8, 60)  
            dummy_x = self.pool(F.relu(self.gn1(self.conv1(dummy_x)))) 
            dummy_x = self.pool(F.relu(self.gn2(self.conv2(dummy_x)))) 
            flattened_dim = dummy_x.view(1, -1).size(1) 
            
        self.fc1 = nn.Linear(flattened_dim, 128) 
        self.dropout = nn.Dropout(dropout_rate) 
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x, return_rep=False):
        x = x.view(-1, 8, 60) 
        x = self.pool(F.relu(self.gn1(self.conv1(x)))) 
        x = self.pool(F.relu(self.gn2(self.conv2(x)))) 
        
        x = torch.flatten(x, 1) 
        rep = F.relu(self.fc1(self.dropout(x))) 
        out = self.fc2(rep) 
        
        if return_rep:
            return out, rep 
        return out 

# ==========================================
# 5. MOON Core & Evaluation Functions
# ==========================================
def train_client_moon(model, global_model, prev_model, dataloader, lr, mu, tau):
    """
    MOON Training: Standard CE loss + Model-Contrastive Loss.
    Maximizes agreement with global representation, minimizes agreement with previous local representation.
    """
    model.train() 
    global_model.eval()
    if prev_model is not None:
        prev_model.eval()
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) 
    cos = nn.CosineSimilarity(dim=-1)
    
    for _ in range(LOCAL_EPOCHS):
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE) 
            optimizer.zero_grad() 
            
            # Current model forward pass
            out, rep = model(data, return_rep=True) 
            loss_sup = criterion(out, target) 
            
            if prev_model is not None:
                # Get representations from frozen global and frozen previous local model
                with torch.no_grad():
                    _, rep_glob = global_model(data, return_rep=True)
                    _, rep_prev = prev_model(data, return_rep=True)
                
                # Compute similarities
                sim_pos = cos(rep, rep_glob)
                sim_neg = cos(rep, rep_prev)
                
                # Reformulate as CrossEntropy for the contrastive task
                # We want the positive pair (global) to be predicted over negative (prev local)
                logits = torch.stack([sim_pos, sim_neg], dim=1) / tau
                labels = torch.zeros(logits.size(0), dtype=torch.long).to(DEVICE)
                
                loss_con = F.cross_entropy(logits, labels)
                loss = loss_sup + (mu * loss_con)
            else:
                # First round: no previous model exists, just use supervised loss
                loss = loss_sup
            
            loss.backward() 
            optimizer.step() 
            
    return model.state_dict() 

def evaluate_model_standard(model, dataloader, num_classes, return_full=False):
    """Standard inference using output logits (MOON does not use prototypes for inference)"""
    model.eval() 
    all_targets, all_preds, all_probs = [], [], [] 

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE) 
            logits = model(data) 
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_targets.extend(target.cpu().numpy()) 
            all_preds.extend(preds.cpu().numpy()) 
            all_probs.extend(probs.cpu().numpy()) 
            
    acc = accuracy_score(all_targets, all_preds) 
    bacc = balanced_accuracy_score(all_targets, all_preds) 
    f1_macro = f1_score(all_targets, all_preds, average='macro') 
    f1_weighted = f1_score(all_targets, all_preds, average='weighted') 
    
    if return_full:
        return 0.0, acc, bacc, f1_macro, f1_weighted, np.array(all_targets), np.array(all_preds), np.array(all_probs) 
    return 0.0, acc, bacc, f1_macro, f1_weighted 

def calculate_calibration_metrics(y_true, y_probs, num_classes):
    nll = log_loss(y_true, y_probs) 
    y_onehot = np.eye(num_classes)[y_true] 
    brier_score = np.mean(np.sum((y_probs - y_onehot)**2, axis=1)) 
    
    n_bins = 10 
    bin_boundaries = np.linspace(0, 1, n_bins + 1) 
    
    def calc_ece(true_binary, probs_1d):
        ece, mce = 0.0, 0.0 
        bin_counts = [] 
        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1] 
            in_bin = (probs_1d > bin_lower) & (probs_1d <= bin_upper) 
            n_samples = np.sum(in_bin) 
            prop_in_bin = n_samples / len(probs_1d) 
            if prop_in_bin > 0:
                accuracy_in_bin = true_binary[in_bin].astype(float).mean() 
                avg_confidence_in_bin = probs_1d[in_bin].mean() 
                diff = np.abs(avg_confidence_in_bin - accuracy_in_bin) 
                ece += prop_in_bin * diff 
                mce = max(mce, diff) 
            bin_counts.append(n_samples) 
        return ece, mce, bin_counts 
        
    confidences = np.max(y_probs, axis=1) 
    predictions = np.argmax(y_probs, axis=1) 
    accuracies = predictions == y_true 
    
    overall_ece, overall_mce, _ = calc_ece(accuracies, confidences) 
    
    per_class_calib = {} 
    for c in range(num_classes):
        c_true = (y_true == c).astype(int) 
        c_probs = y_probs[:, c] 
        c_ece, c_mce, c_bins = calc_ece(c_true, c_probs) 
        per_class_calib[class_names[c]] = {'ECE': c_ece, 'MCE': c_mce, 'Samples_Per_Bin': c_bins} 
            
    return overall_ece, overall_mce, brier_score, nll, per_class_calib 

# ==========================================
# 6. Helper Functions for Evaluation
# ==========================================
def get_representations(model, dataloader):
    model.eval() 
    reps, labels = [], [] 
    with torch.no_grad():
        for data, target in dataloader:
            _, rep = model(data.to(DEVICE), return_rep=True) 
            reps.extend(rep.cpu().numpy()) 
            labels.extend(target.numpy()) 
    return np.array(reps), np.array(labels) 

def generate_tsne_plots(global_model, top_local_models, round_num):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12)) 
    fig.suptitle(f't-SNE Embeddings at Round {round_num} (MOON)', fontsize=16) 
    
    g_reps, g_lbls = get_representations(global_model, val_loader) 
    g_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(g_reps) 
    axes[0, 0].scatter(g_tsne[:, 0], g_tsne[:, 1], c=g_lbls, cmap='tab10', alpha=0.7) 
    axes[0, 0].set_title('Global Model (Shared Validation Data)') 
    
    positions = [(0, 1), (1, 0), (1, 1)] 
    for idx, cid in enumerate(top_3_clients):
        if cid in top_local_models:
            l_reps, l_lbls = get_representations(top_local_models[cid], client_loaders[cid]) 
            l_tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(l_reps)-1)), random_state=42).fit_transform(l_reps) 
            axes[positions[idx][0], positions[idx][1]].scatter(l_tsne[:, 0], l_tsne[:, 1], c=l_lbls, cmap='tab10', alpha=0.7) 
            axes[positions[idx][0], positions[idx][1]].set_title(f'Local Model - Client {cid} (Local Train Data)') 
    
    fig.text(0.5, 0.01, "Note: t-SNE coordinates are fitted independently.", ha='center', fontsize=10, style='italic')     
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(f'visualizations/tsne_round_{round_num}_moon.png') 
    plt.close() 

# ==========================================
# 7. Main Execution (MOON)
# ==========================================
global_model = OptimizedSensorCNN(num_classes, best_params['filter1'], best_params['filter2'], best_params['dropout_rate'], best_params['kernel_size']).to(DEVICE) 
history = {'round': [], 'acc': [], 'balanced_acc': [], 'f1_macro': [], 'f1_weighted': []} 

best_val_f1 = 0.0
patience_counter = 0 
best_global_state = copy.deepcopy(global_model.state_dict()) 

# Dictionary to hold the local models from the PREVIOUS round for each client
prev_local_models = {}

for round_num in range(1, MAIN_FL_ROUNDS + 1):
    print(f"\n--- Starting Main FL Round {round_num}/{MAIN_FL_ROUNDS} (MOON) ---") 
    current_lr = best_params['lr'] * (0.98 ** (round_num - 1)) 
    local_weights, client_sizes = [], [] 
    top_3_models = {} 
    
    for cid in tqdm(active_client_ids, desc=f"Training All Clients (LR: {current_lr:.5f})"):
        # 1. Initialize client model with global weights
        client_model = copy.deepcopy(global_model) 
        
        # 2. Fetch the client's previous local model (if available)
        prev_model = prev_local_models.get(cid, None)
        
        # 3. Train using MOON objective
        weights = train_client_moon(client_model, global_model, prev_model, client_loaders[cid], current_lr, best_params['mu'], best_params['tau']) 
        
        # 4. Save the trained model for NEXT round's contrastive loss
        prev_local_models[cid] = copy.deepcopy(client_model)
        
        local_weights.append(weights) 
        client_sizes.append(len(client_loaders[cid].dataset)) 
        
        if cid in top_3_clients:
            top_3_models[cid] = client_model 
            
    # Aggregate weights normally (FedAvg style)
    global_state = weighted_fedavg(local_weights, client_sizes, global_model.state_dict()) 
    global_model.load_state_dict(global_state) 
    
    # Evaluate (Standard Output Inference, no prototypes needed)
    _, val_acc, val_bacc, val_f1_macro, val_f1_weighted = evaluate_model_standard(global_model, val_loader, num_classes) 
    history['round'].append(round_num) 
    history['acc'].append(val_acc) 
    history['balanced_acc'].append(val_bacc) 
    history['f1_macro'].append(val_f1_macro) 
    history['f1_weighted'].append(val_f1_weighted) 
    
    print(f"Validation -> Acc: {val_acc:.4f} | B-Acc: {val_bacc:.4f} | F1-Macro: {val_f1_macro:.4f}") 
    print("Generating t-SNE projections...") 
    generate_tsne_plots(global_model, top_3_models, round_num) 

    if val_f1_macro > best_val_f1:
        best_val_f1 = val_f1_macro 
        patience_counter = 0 
        best_global_state = copy.deepcopy(global_model.state_dict()) 
    else:
        patience_counter += 1 
        print(f"Early Stopping Counter: {patience_counter}/{PATIENCE}") 
        
    # if patience_counter >= PATIENCE:
    #     print(f"\n[!] Validation F1-Macro hasn't improved for {PATIENCE} rounds. Early stopping triggered!") 
    #     break 

# Restore best global state before final evaluation
print("Restoring best model weights for final evaluation...") 
global_model.load_state_dict(best_global_state) 

# --- Final Evaluation ---
_, test_acc, test_bacc, test_f1_macro, test_f1_weighted, y_true, calibrated_pred, test_probs = evaluate_model_standard(
    global_model, test_loader, num_classes, return_full=True
)

print(f"\n=== Final Global Model Test Metrics (MOON Standard Inference) ===") 
print(f"Accuracy: {test_acc:.4f} | Macro-F1: {test_f1_macro:.4f} | Weighted-F1: {test_f1_weighted:.4f}") 

ece_a, mce_a, brier_a, nll_a, per_class_calib_a = calculate_calibration_metrics(y_true, test_probs, num_classes) 

print("\n=== Per-Class Calibration (Standard Output) ===") 
for c_name, mets in per_class_calib_a.items():
    print(f"Class {c_name} -> ECE: {mets['ECE']:.4f}, MCE: {mets['MCE']:.4f}, Samples/Bin: {mets['Samples_Per_Bin']}") 

cm = confusion_matrix(y_true, calibrated_pred) 
print("\n=== Detailed Classification Report (MOON) ===") 
print(classification_report(y_true, calibrated_pred, target_names=class_names)) 

# --- File Saves ---
report_dict = classification_report(y_true, calibrated_pred, target_names=class_names, output_dict=True) 
report_df = pd.DataFrame(report_dict).transpose() 
history_df = pd.DataFrame(history) 

calibration_df = pd.DataFrame({
    'Metric': ['ECE', 'MCE', 'Brier Score', 'NLL'],
    'Value': [ece_a, mce_a, brier_a, nll_a]
}) 

torch.save(global_model.state_dict(), 'models/final_global_model_moon.pth') 

with pd.ExcelWriter('reports/FL_Final_Reports_MOON.xlsx') as writer:
    report_df.to_excel(writer, sheet_name='Classification_Report') 
    history_df.to_excel(writer, sheet_name='Round_Metrics', index=False) 
    calibration_df.to_excel(writer, sheet_name='Calibration_Metrics', index=False) 

plt.figure(figsize=(8, 6)) 
sns.heatmap(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names) 
plt.title('Normalized Confusion Matrix (MOON)') 
plt.ylabel('True Class') 
plt.xlabel('Predicted Class') 
plt.tight_layout() 
plt.savefig('visualizations/confusion_matrix_moon.png') 
plt.close() 

plt.figure(figsize=(10, 6)) 
plt.plot(history['round'], history['acc'], label='Accuracy', marker='o', linewidth=2) 
plt.plot(history['round'], history['balanced_acc'], label='Balanced Acc', marker='^', linestyle='--', linewidth=2) 
plt.plot(history['round'], history['f1_macro'], label='Macro F1', marker='s', linewidth=2) 
plt.title('Global Model: Validation Metrics (MOON Inference)', fontsize=14) 
plt.xlabel('Federated Round', fontsize=12) 
plt.ylabel('Score', fontsize=12) 
plt.legend() 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.tight_layout() 
plt.savefig('visualizations/learning_curves_moon.png') 
plt.close() 

plt.figure(figsize=(8, 6)) 
plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated') 
for i in range(num_classes):
    true_binary = (y_true == i).astype(int) 
    prob_true, prob_pred = calibration_curve(true_binary, test_probs[:, i], n_bins=10) 
    plt.plot(prob_pred, prob_true, marker='o', label=f'Class {class_names[i]}') 

plt.title('Reliability Diagram (MOON Standard)', fontsize=14) 
plt.xlabel('Mean Predicted Probability') 
plt.ylabel('Fraction of Positives') 
plt.legend() 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.tight_layout() 
plt.savefig('visualizations/calibration_curve_moon.png') 
plt.close()

print("\nExecution complete.")