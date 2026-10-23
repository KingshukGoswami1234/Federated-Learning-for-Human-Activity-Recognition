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
import optuna

# ==========================================
# 1. Configuration & Setup
# ==========================================
MAIN_FL_ROUNDS = 25 
LOCAL_EPOCHS = 10     
BATCH_SIZE = 32
HPO_TRIALS = 10    
HPO_ROUNDS = 5       
PATIENCE = 5 

TARGET_CLASSES = ['A', 'B', 'D', 'E'] 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(f"Executing on device: {DEVICE}") 

os.makedirs('visualizations', exist_ok=True) 
os.makedirs('reports', exist_ok=True) 
os.makedirs('models', exist_ok=True) 

os.makedirs('visualizations_baseline', exist_ok=True)
os.makedirs('reports_baseline', exist_ok=True)
os.makedirs('models_baseline', exist_ok=True)

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
    """Sample-weighted FedAvg aggregation."""
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
# 5. FedAvg Core & Evaluation Functions
# ==========================================
def train_client_fedavg(model, dataloader, lr):
    model.train() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) 
    for _ in range(LOCAL_EPOCHS):
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE) 
            optimizer.zero_grad() 
            output = model(data) 
            loss = criterion(output, target) 
            loss.backward() 
            optimizer.step() 
    return model.state_dict() 

def evaluate_model(model, dataloader, return_full=False):
    model.eval() 
    all_targets, all_preds, all_probs = [], [], [] 
    total_loss = 0.0 
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE) 
            output = model(data) 
            total_loss += F.cross_entropy(output, target).item() * data.size(0) 
            
            probs = F.softmax(output, dim=1) 
            preds = output.argmax(dim=1) 
            
            all_targets.extend(target.cpu().numpy()) 
            all_preds.extend(preds.cpu().numpy()) 
            all_probs.extend(probs.cpu().numpy()) 
            
    avg_loss = total_loss / len(dataloader.dataset) 
    acc = accuracy_score(all_targets, all_preds) 
    bacc = balanced_accuracy_score(all_targets, all_preds) 
    f1_macro = f1_score(all_targets, all_preds, average='macro') 
    f1_weighted = f1_score(all_targets, all_preds, average='weighted') 
    
    if return_full:
        return avg_loss, acc, bacc, f1_macro, f1_weighted, np.array(all_targets), np.array(all_preds), np.array(all_probs) 
    return avg_loss, acc, bacc, f1_macro, f1_weighted 

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
# 6. Optuna Hyperparameter Optimization
# ==========================================
def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True) 
    filter1 = trial.suggest_categorical("filter1", [64, 128]) 
    filter2 = trial.suggest_categorical("filter2", [64, 128]) 
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4) 
    kernel_size = trial.suggest_categorical("kernel_size", [5, 7, 9]) 

    print(f"\n[Optuna] Starting Trial {trial.number + 1}/{HPO_TRIALS}") 
    print(f"         Testing Params: LR: {lr:.5f}, Filters: {filter1}/{filter2}, Drop: {dropout_rate:.2f}, Kernel: {kernel_size}") 
    
    model = OptimizedSensorCNN(num_classes, filter1, filter2, dropout_rate, kernel_size).to(DEVICE) 
    
    for round_num in range(HPO_ROUNDS):
        print(f"         > HPO Round {round_num + 1}/{HPO_ROUNDS} running...") 
        local_weights = [] 
        client_sizes = [] 
        
        for cid in active_client_ids:
            client_model = copy.deepcopy(model) 
            weights = train_client_fedavg(client_model, client_loaders[cid], lr) 
            local_weights.append(weights) 
            client_sizes.append(len(client_loaders[cid].dataset)) 
            
        global_state = weighted_fedavg(local_weights, client_sizes, model.state_dict()) 
        model.load_state_dict(global_state) 
        
    _, _, _, val_f1_macro, _ = evaluate_model(model, val_loader) 
    return val_f1_macro 

print("\n--- Starting Optuna Hyperparameter Search (Maximizing Macro-F1) ---") 
optuna.logging.set_verbosity(optuna.logging.WARNING)  
study = optuna.create_study(direction="maximize") 
study.optimize(objective, n_trials=HPO_TRIALS) 
best_params = study.best_params 
print(f"\n[Optuna Search Complete] Best Parameters: {best_params}") 

# ==========================================
# 7. Helper Functions for Evaluation
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

def collect_logits(model, dataloader):
    model.eval() 
    all_logits, all_targets = [], [] 
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data.to(DEVICE)) 
            all_logits.append(output.cpu()) 
            all_targets.append(target.cpu()) 
    return torch.cat(all_logits), torch.cat(all_targets) 

def fit_temperature(val_logits, val_targets):
    val_logits = val_logits.to(DEVICE) 
    val_targets = val_targets.to(DEVICE) 
    log_temperature = torch.nn.Parameter(torch.zeros(1, device=DEVICE)) 
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=100) 
    def closure():
        optimizer.zero_grad() 
        loss = F.cross_entropy(val_logits / torch.exp(log_temperature), val_targets) 
        loss.backward() 
        return loss 
    optimizer.step(closure) 
    return max(torch.exp(log_temperature).item(), 1e-6) 

def softmax_numpy(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float64) / temperature 
    logits = logits - np.max(logits, axis=1, keepdims=True) 
    exp_logits = np.exp(logits) 
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True) 

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
# 8. Main Execution
# ==========================================
global_model = OptimizedSensorCNN(num_classes, best_params['filter1'], best_params['filter2'], best_params['dropout_rate'], best_params['kernel_size']).to(DEVICE) 
history = {'round': [], 'loss': [], 'acc': [], 'balanced_acc': [], 'f1_macro': [], 'f1_weighted': []} 

best_val_loss = float('inf') 
patience_counter = 0 
best_global_state = copy.deepcopy(global_model.state_dict()) 

for round_num in range(1, MAIN_FL_ROUNDS + 1):
    print(f"\n--- Starting Main FL Round {round_num}/{MAIN_FL_ROUNDS} ---") 
    current_lr = best_params['lr'] * (0.98 ** (round_num - 1)) 
    local_weights, client_sizes = [], [] 
    top_3_models = {} 
    
    for cid in tqdm(active_client_ids, desc=f"Training All Clients (LR: {current_lr:.5f})"):
        client_model = copy.deepcopy(global_model) 
        weights = train_client_fedavg(client_model, client_loaders[cid], current_lr) 
        local_weights.append(weights) 
        client_sizes.append(len(client_loaders[cid].dataset)) 
        if cid in top_3_clients:
            top_3_models[cid] = client_model 
            
    global_state = weighted_fedavg(local_weights, client_sizes, global_model.state_dict()) 
    global_model.load_state_dict(global_state) 
    
    val_loss, val_acc, val_bacc, val_f1_macro, val_f1_weighted = evaluate_model(global_model, val_loader) 
    history['round'].append(round_num) 
    history['loss'].append(val_loss) 
    history['acc'].append(val_acc) 
    history['balanced_acc'].append(val_bacc) 
    history['f1_macro'].append(val_f1_macro) 
    history['f1_weighted'].append(val_f1_weighted) 
    
    print(f"Validation -> Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | B-Acc: {val_bacc:.4f} | F1-Macro: {val_f1_macro:.4f}") 
    
    print("Generating t-SNE projections...") 
    generate_tsne_plots(global_model, top_3_models, round_num) 

    if val_loss < best_val_loss:
        best_val_loss = val_loss 
        patience_counter = 0 
        best_global_state = copy.deepcopy(global_model.state_dict()) 
    else:
        patience_counter += 1 
        print(f"Early Stopping Counter: {patience_counter}/{PATIENCE}") 
        
    if patience_counter >= PATIENCE:
        print(f"\n[!] Validation loss hasn't improved for {PATIENCE} rounds. Early stopping triggered!") 
        break 

# Restore best global state before final evaluation
print("Restoring best model weights for final evaluation...") 
global_model.load_state_dict(best_global_state) 

# --- Final Evaluation + Temperature Scaling ---
test_logits, test_targets_t = collect_logits(global_model, test_loader) 
test_logits_np, y_true = test_logits.numpy(), test_targets_t.numpy() 
uncalibrated_probs = softmax_numpy(test_logits_np, temperature=1.0) 

val_logits, val_targets_t = collect_logits(global_model, val_loader) 
temperature = fit_temperature(val_logits, val_targets_t) 

calibrated_probs = softmax_numpy(test_logits_np, temperature=temperature) 
calibrated_pred = np.argmax(calibrated_probs, axis=1) 

print(f"\n=== Temperature Scaling ===") 
print(f"Learned temperature T: {temperature:.6f}") 

test_acc = accuracy_score(y_true, calibrated_pred) 
test_bacc = balanced_accuracy_score(y_true, calibrated_pred) 
test_f1_macro = f1_score(y_true, calibrated_pred, average='macro') 
test_f1_weighted = f1_score(y_true, calibrated_pred, average='weighted') 

ece_b, mce_b, brier_b, nll_b, per_class_calib_b = calculate_calibration_metrics(y_true, uncalibrated_probs, num_classes) 
ece_a, mce_a, brier_a, nll_a, per_class_calib_a = calculate_calibration_metrics(y_true, calibrated_probs, num_classes) 

print(f"\n=== Final Global Model Test Metrics ===") 
print(f"Accuracy: {test_acc:.4f} | Macro-F1: {test_f1_macro:.4f} | Weighted-F1: {test_f1_weighted:.4f}") 

print(f"\n=== Calibration: Before vs After Temperature Scaling ===") 
print(f"Overall ECE  : {ece_b:.4f} -> {ece_a:.4f}") 
print(f"Overall MCE  : {mce_b:.4f} -> {mce_a:.4f}") 

print("\n=== Per-Class Calibration (After Temp Scaling) ===") 
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

print("\n=== Detailed Classification Report (Calibrated) ===") 
print(classification_report(y_true, calibrated_pred, target_names=class_names)) 

# --- File Saves ---
report_dict = classification_report(y_true, calibrated_pred, target_names=class_names, output_dict=True) 
report_df = pd.DataFrame(report_dict).transpose() 
history_df = pd.DataFrame(history) 

calibration_df = pd.DataFrame({
    'Metric': ['ECE', 'MCE', 'Brier Score', 'NLL'],
    'Before_Temperature_Scaling': [ece_b, mce_b, brier_b, nll_b],
    'After_Temperature_Scaling': [ece_a, mce_a, brier_a, nll_a]
}) 

torch.save(global_model.state_dict(), 'models/final_global_model.pth') 
with open('models/temperature_scaling.txt', 'w') as f:
    f.write(f'Temperature={temperature:.10f}\n') 

with pd.ExcelWriter('reports/FL_Final_Reports.xlsx') as writer:
    report_df.to_excel(writer, sheet_name='Classification_Report') 
    history_df.to_excel(writer, sheet_name='Round_Metrics', index=False) 
    calibration_df.to_excel(writer, sheet_name='Calibration_Before_After', index=False) 
    prototype_df.to_excel(writer, sheet_name='Prototype_AB_DE', index=False) 

plt.figure(figsize=(8, 6)) 
sns.heatmap(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names) 
plt.title('Normalized Confusion Matrix') 
plt.ylabel('True Class') 
plt.xlabel('Predicted Class') 
plt.tight_layout() 
plt.savefig('visualizations/confusion_matrix.png') 
plt.close() 

fig, axes = plt.subplots(1, 2, figsize=(16, 6)) 
for ax, probs, title in [(axes[0], uncalibrated_probs, 'Before Temperature Scaling'), (axes[1], calibrated_probs, 'After Temperature Scaling')]: 
    ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated') 
    for i in range(num_classes):
        true_binary = (y_true == i).astype(int) 
        prob_true, prob_pred = calibration_curve(true_binary, probs[:, i], n_bins=10) 
        ax.plot(prob_pred, prob_true, marker='o', label=f'Class {class_names[i]}') 
    ax.set_title(title) 
    ax.set_xlabel('Mean Predicted Probability') 
    ax.set_ylabel('Fraction of Positives') 
    ax.legend() 
    ax.grid(True) 
fig.suptitle(f'Reliability Diagram | T = {temperature:.3f}', fontsize=15) 
plt.tight_layout() 
plt.savefig('visualizations/calibration_curve_before_after.png') 
plt.close() 

plt.figure(figsize=(16, 6)) 
plt.subplot(1, 2, 1) 
plt.plot(history['round'], history['loss'], label='Validation Loss', color='red', marker='o', linewidth=2) 
plt.title('Global Model: Validation Loss', fontsize=14) 
plt.xlabel('Federated Round', fontsize=12) 
plt.ylabel('Cross-Entropy Loss', fontsize=12) 
plt.legend() 
plt.grid(True, linestyle='--', alpha=0.7) 

plt.subplot(1, 2, 2) 
plt.plot(history['round'], history['acc'], label='Accuracy', marker='o', linewidth=2) 
plt.plot(history['round'], history['balanced_acc'], label='Balanced Acc', marker='^', linestyle='--', linewidth=2) 
plt.plot(history['round'], history['f1_macro'], label='Macro F1', marker='s', linewidth=2) 
plt.title('Global Model: Validation Metrics', fontsize=14) 
plt.xlabel('Federated Round', fontsize=12) 
plt.ylabel('Score', fontsize=12) 
plt.legend() 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.tight_layout() 
plt.savefig('visualizations/learning_curves.png') 
plt.close() 


# ==========================================
# 9. Centralized Learning Baseline
# ==========================================
print("\n--- Starting Centralized Learning Baseline ---")

cent_model = OptimizedSensorCNN(num_classes, best_params['filter1'], best_params['filter2'], best_params['dropout_rate'], best_params['kernel_size']).to(DEVICE)
cent_optimizer = torch.optim.AdamW(cent_model.parameters(), lr=best_params['lr'], weight_decay=1e-4)

cent_best_val_loss = float('inf')
cent_patience_counter = 0
cent_best_state = copy.deepcopy(cent_model.state_dict())
cent_history = {'epoch': [], 'loss': [], 'acc': [], 'balanced_acc': [], 'f1_macro': [], 'f1_weighted': []}

for epoch in range(1, MAIN_FL_ROUNDS + 1):
    cent_model.train()
    for data, target in tqdm(full_train_loader, desc=f"Centralized Epoch {epoch}/{MAIN_FL_ROUNDS}"):
        data, target = data.to(DEVICE), target.to(DEVICE)
        cent_optimizer.zero_grad()
        output = cent_model(data)
        loss = criterion(output, target)
        loss.backward()
        cent_optimizer.step()

    val_loss, val_acc, val_bacc, val_f1_macro, val_f1_weighted = evaluate_model(cent_model, val_loader)
    cent_history['epoch'].append(epoch)
    cent_history['loss'].append(val_loss)
    cent_history['acc'].append(val_acc)
    cent_history['balanced_acc'].append(val_bacc)
    cent_history['f1_macro'].append(val_f1_macro)
    cent_history['f1_weighted'].append(val_f1_weighted)

    print(f"Cent. Validation -> Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | B-Acc: {val_bacc:.4f} | F1-Macro: {val_f1_macro:.4f}")

    if val_loss < cent_best_val_loss:
        cent_best_val_loss = val_loss
        cent_patience_counter = 0
        cent_best_state = copy.deepcopy(cent_model.state_dict())
    else:
        cent_patience_counter += 1
        print(f"Centralized Early Stopping Counter: {cent_patience_counter}/{PATIENCE}")
        
    if cent_patience_counter >= PATIENCE:
        print(f"\n[!] Centralized validation loss hasn't improved for {PATIENCE} epochs. Early stopping triggered!")
        break

print("Restoring best centralized model weights for final evaluation...")
cent_model.load_state_dict(cent_best_state)

c_test_logits, c_test_targets_t = collect_logits(cent_model, test_loader)
c_test_logits_np = c_test_logits.numpy()

c_val_logits, c_val_targets_t = collect_logits(cent_model, val_loader)
c_temperature = fit_temperature(c_val_logits, c_val_targets_t)

c_calibrated_probs = softmax_numpy(c_test_logits_np, temperature=c_temperature)
c_calibrated_pred = np.argmax(c_calibrated_probs, axis=1)

c_test_acc = accuracy_score(y_true, c_calibrated_pred)
c_test_bacc = balanced_accuracy_score(y_true, c_calibrated_pred)
c_test_f1_macro = f1_score(y_true, c_calibrated_pred, average='macro')
c_test_f1_weighted = f1_score(y_true, c_calibrated_pred, average='weighted')

print(f"\n=== Final Centralized Model Test Metrics ===")
print(f"Accuracy: {c_test_acc:.4f} | Macro-F1: {c_test_f1_macro:.4f} | Weighted-F1: {c_test_f1_weighted:.4f}")

# --- File Saves for Centralized Baseline ---
c_report_dict = classification_report(y_true, c_calibrated_pred, target_names=class_names, output_dict=True)
c_report_df = pd.DataFrame(c_report_dict).transpose()
c_history_df = pd.DataFrame(cent_history)

torch.save(cent_model.state_dict(), 'models_baseline/final_centralized_model.pth')
with open('models_baseline/temperature_scaling.txt', 'w') as f:
    f.write(f'Temperature={c_temperature:.10f}\n')

with pd.ExcelWriter('reports_baseline/Centralized_Final_Reports.xlsx') as writer:
    c_report_df.to_excel(writer, sheet_name='Classification_Report')
    c_history_df.to_excel(writer, sheet_name='Epoch_Metrics', index=False)

# ==========================================
# 10. Performance Comparison
# ==========================================
print("\n==========================================")
print("10. Side-by-Side Performance Comparison")
print("==========================================")
comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Balanced Accuracy', 'F1 Macro', 'F1 Weighted'],
    'FL Global Model': [test_acc, test_bacc, test_f1_macro, test_f1_weighted],
    'Centralized Baseline': [c_test_acc, c_test_bacc, c_test_f1_macro, c_test_f1_weighted]
})
print(comparison_df.to_string(index=False))
comparison_df.to_excel('reports_baseline/FL_vs_Centralized_Comparison.xlsx', index=False)

print("\nExecution complete.")