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

# Hardcoded best parameters
best_params = {
    'lr': 0.004160308766542998, 
    'mu': 0.010115889610221198, 
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

print(f"Total Active Clients (with 4-class data): {len(active_client_ids)} (100% Participation)") 

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
        self.fc2 = nn.Linear(128, num_classes) # Kept for backward propagation during training
        
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
# 5. ProtoFed Core & Evaluation Functions
# ==========================================
def compute_local_prototypes(model, dataloader, num_classes):
    """Computes the local average feature representations (prototypes) per class."""
    model.eval()
    class_sums = {i: torch.zeros(128).to(DEVICE) for i in range(num_classes)}
    class_counts = {i: 0 for i in range(num_classes)}
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            _, rep = model(data, return_rep=True)
            for i in range(num_classes):
                mask = (target == i)
                if mask.sum() > 0:
                    class_sums[i] += rep[mask].sum(dim=0)
                    class_counts[i] += mask.sum().item()
                    
    local_prototypes = {}
    for i in range(num_classes):
        if class_counts[i] > 0:
            local_prototypes[i] = class_sums[i] / class_counts[i]
    return local_prototypes

def aggregate_global_prototypes(client_prototypes_list, num_classes):
    """Averages prototypes across clients that have data for that class."""
    global_prototypes = {}
    for i in range(num_classes):
        protos = [cp[i] for cp in client_prototypes_list if i in cp]
        if len(protos) > 0:
            global_prototypes[i] = torch.stack(protos).mean(dim=0)
    return global_prototypes

def train_client_fedprox(model, global_model_weights, dataloader, lr, mu):
    model.train() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) 
    for _ in range(LOCAL_EPOCHS):
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE) 
            optimizer.zero_grad() 
            output = model(data) 
            loss = criterion(output, target) 
            
            proximal_term = 0.0 
            for w, w_t in zip(model.parameters(), global_model_weights):
                proximal_term += (w - w_t).pow(2).sum() 
            loss += (mu / 2) * proximal_term 
            
            loss.backward() 
            optimizer.step() 
    return model.state_dict() 

def evaluate_model_proto(model, dataloader, global_prototypes, num_classes, return_full=False):
    """Inference based on L2 distance to global prototypes as per ProtoFed."""
    model.eval() 
    all_targets, all_preds, all_dists = [], [], [] 
    
    # Create prototype tensor
    proto_tensor = torch.zeros(num_classes, 128).to(DEVICE)
    for i in range(num_classes):
        if i in global_prototypes:
            proto_tensor[i] = global_prototypes[i]

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE) 
            _, rep = model(data, return_rep=True) 
            
            # Compute L2 distances from representations to each global prototype
            # shape: (batch_size, num_classes)
            dists = torch.cdist(rep, proto_tensor, p=2.0)
            preds = torch.argmin(dists, dim=1)
            
            all_targets.extend(target.cpu().numpy()) 
            all_preds.extend(preds.cpu().numpy()) 
            # Convert distances to pseudo-probabilities via softmax of negative distance for calibration metrics
            probs = F.softmax(-dists, dim=1)
            all_dists.extend(probs.cpu().numpy()) 
            
    acc = accuracy_score(all_targets, all_preds) 
    bacc = balanced_accuracy_score(all_targets, all_preds) 
    f1_macro = f1_score(all_targets, all_preds, average='macro') 
    f1_weighted = f1_score(all_targets, all_preds, average='weighted') 
    
    if return_full:
        return 0.0, acc, bacc, f1_macro, f1_weighted, np.array(all_targets), np.array(all_preds), np.array(all_dists) 
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
    fig.suptitle(f't-SNE Embeddings at Round {round_num}', fontsize=16) 
    
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
    
    fig.text(0.5, 0.01, "Note: t-SNE coordinates are fitted independently and are not directly comparable across plots/rounds.", ha='center', fontsize=10, style='italic')     
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(f'visualizations/tsne_round_{round_num}.png') 
    plt.close() 

def calculate_prototype_overlap(reps, labels, class_names):
    prototypes = {} 
    for class_idx, class_name in enumerate(class_names):
        class_reps = reps[labels == class_idx] 
        if len(class_reps) > 0:
            prototypes[class_name] = class_reps.mean(axis=0) 

    rows = [] 
    for c1, c2 in [('A', 'B'), ('D', 'E')]:
        if c1 not in prototypes or c2 not in prototypes:
            continue 
        p1, p2 = prototypes[c1], prototypes[c2] 
        p1_norm, p2_norm = np.linalg.norm(p1), np.linalg.norm(p2) 
        
        mask1, mask2 = labels == class_names.index(c1), labels == class_names.index(c2) 
        r1, r2 = reps[mask1], reps[mask2] 
        d11, d12 = np.linalg.norm(r1 - p1, axis=1), np.linalg.norm(r1 - p2, axis=1) 
        d22, d21 = np.linalg.norm(r2 - p2, axis=1), np.linalg.norm(r2 - p1, axis=1) 

        rows.append({
            'Pair': f'{c1}-{c2}',
            'Prototype_Euclidean_Distance': float(np.linalg.norm(p1 - p2)),
            'Prototype_Cosine_Similarity': float(np.dot(p1, p2) / (p1_norm * p2_norm + 1e-12)),
            f'{c1}_to_{c2}_Cross_Assignment_Rate': float(np.mean(d12 < d11)),
            f'{c2}_to_{c1}_Cross_Assignment_Rate': float(np.mean(d21 < d22)),
            'Symmetric_Prototype_Overlap': (float(np.mean(d12 < d11)) + float(np.mean(d21 < d22))) / 2.0
        }) 
    return pd.DataFrame(rows) 

# ==========================================
# 7. Main Execution
# ==========================================
global_model = OptimizedSensorCNN(num_classes, best_params['filter1'], best_params['filter2'], best_params['dropout_rate'], best_params['kernel_size']).to(DEVICE) 
history = {'round': [], 'acc': [], 'balanced_acc': [], 'f1_macro': [], 'f1_weighted': []} 

best_val_f1 = 0.0
patience_counter = 0 
best_global_state = copy.deepcopy(global_model.state_dict()) 
best_global_prototypes = None

for round_num in range(1, MAIN_FL_ROUNDS + 1):
    print(f"\n--- Starting Main FL Round {round_num}/{MAIN_FL_ROUNDS} ---") 
    current_lr = best_params['lr'] * (0.98 ** (round_num - 1)) 
    local_weights, client_sizes, client_prototypes_list = [], [], [] 
    top_3_models = {} 
    global_weights_frozen = [p.clone().detach() for p in global_model.parameters()] 
    
    for cid in tqdm(active_client_ids, desc=f"Training All Clients (LR: {current_lr:.5f})"):
        client_model = copy.deepcopy(global_model) 
        weights = train_client_fedprox(client_model, global_weights_frozen, client_loaders[cid], current_lr, best_params['mu']) 
        local_weights.append(weights) 
        client_sizes.append(len(client_loaders[cid].dataset)) 
        local_protos = compute_local_prototypes(client_model, client_loaders[cid], num_classes)
        client_prototypes_list.append(local_protos)

        if cid in top_3_clients:
            top_3_models[cid] = client_model 
            
    global_state = weighted_fedavg(local_weights, client_sizes, global_model.state_dict()) 
    global_model.load_state_dict(global_state) 
    global_prototypes = aggregate_global_prototypes(client_prototypes_list, num_classes)
    
    _, val_acc, val_bacc, val_f1_macro, val_f1_weighted = evaluate_model_proto(global_model, val_loader, global_prototypes, num_classes) 
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
        best_global_prototypes = copy.deepcopy(global_prototypes)
    else:
        patience_counter += 1 
        print(f"Early Stopping Counter: {patience_counter}/{PATIENCE}") 
        
    if patience_counter >= PATIENCE:
        print(f"\n[!] Validation F1-Macro hasn't improved for {PATIENCE} rounds. Early stopping triggered!") 
        break 

# Restore best global state before final evaluation
print("Restoring best model weights and prototypes for final evaluation...") 
global_model.load_state_dict(best_global_state) 
global_prototypes = best_global_prototypes

# --- Final Evaluation ---
_, test_acc, test_bacc, test_f1_macro, test_f1_weighted, y_true, calibrated_pred, test_probs = evaluate_model_proto(
    global_model, test_loader, global_prototypes, num_classes, return_full=True
)

print(f"\n=== Final Global Model Test Metrics (ProtoFed L2 Inference) ===") 
print(f"Accuracy: {test_acc:.4f} | Macro-F1: {test_f1_macro:.4f} | Weighted-F1: {test_f1_weighted:.4f}") 

ece_a, mce_a, brier_a, nll_a, per_class_calib_a = calculate_calibration_metrics(y_true, test_probs, num_classes) 

print("\n=== Per-Class Calibration (Distance-Softmax) ===") 
for c_name, mets in per_class_calib_a.items():
    print(f"Class {c_name} -> ECE: {mets['ECE']:.4f}, MCE: {mets['MCE']:.4f}, Samples/Bin: {mets['Samples_Per_Bin']}") 

# --- Error & Prototype Analysis ---
final_reps, final_rep_labels = get_representations(global_model, val_loader) 
prototype_df = calculate_prototype_overlap(final_reps, final_rep_labels, class_names) 

cm = confusion_matrix(y_true, calibrated_pred) 
a_idx, b_idx = class_names.index('A'), class_names.index('B') 
e_idx, d_idx = class_names.index('E'), class_names.index('D') 
print("\n=== Class-Specific Error Analysis ===") 
print(f"A -> B Errors: {cm[a_idx, b_idx]} / {cm[a_idx].sum()} ({(cm[a_idx, b_idx]/cm[a_idx].sum())*100:.2f}%)") 
print(f"E -> D Errors: {cm[e_idx, d_idx]} / {cm[e_idx].sum()} ({(cm[e_idx, d_idx]/cm[e_idx].sum())*100:.2f}%)") 
print("\n=== A/B and D/E Prototype Analysis (Validation Representations) ===") 
if len(prototype_df) > 0: print(prototype_df.to_string(index=False)) 

print("\n=== Detailed Classification Report (ProtoFed) ===") 
print(classification_report(y_true, calibrated_pred, target_names=class_names)) 

# --- File Saves ---
report_dict = classification_report(y_true, calibrated_pred, target_names=class_names, output_dict=True) 
report_df = pd.DataFrame(report_dict).transpose() 
history_df = pd.DataFrame(history) 

calibration_df = pd.DataFrame({
    'Metric': ['ECE', 'MCE', 'Brier Score', 'NLL'],
    'Value': [ece_a, mce_a, brier_a, nll_a]
}) 

torch.save(global_model.state_dict(), 'models/final_global_model.pth') 
torch.save(global_prototypes, 'models/global_prototypes.pth')

with pd.ExcelWriter('reports/FL_Final_Reports.xlsx') as writer:
    report_df.to_excel(writer, sheet_name='Classification_Report') 
    history_df.to_excel(writer, sheet_name='Round_Metrics', index=False) 
    calibration_df.to_excel(writer, sheet_name='Calibration_Metrics', index=False) 
    prototype_df.to_excel(writer, sheet_name='Prototype_AB_DE', index=False) 

plt.figure(figsize=(8, 6)) 
sns.heatmap(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names) 
plt.title('Normalized Confusion Matrix (ProtoFed)') 
plt.ylabel('True Class') 
plt.xlabel('Predicted Class') 
plt.tight_layout() 
plt.savefig('visualizations/confusion_matrix.png') 
plt.close() 

plt.figure(figsize=(10, 6)) 
plt.plot(history['round'], history['acc'], label='Accuracy', marker='o', linewidth=2) 
plt.plot(history['round'], history['balanced_acc'], label='Balanced Acc', marker='^', linestyle='--', linewidth=2) 
plt.plot(history['round'], history['f1_macro'], label='Macro F1', marker='s', linewidth=2) 
plt.title('Global Model: Validation Metrics (ProtoFed Inference)', fontsize=14) 
plt.xlabel('Federated Round', fontsize=12) 
plt.ylabel('Score', fontsize=12) 
plt.legend() 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.tight_layout() 
plt.savefig('visualizations/learning_curves.png') 
plt.close() 

print("\nExecution complete.")