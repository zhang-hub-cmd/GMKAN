from model import BKAN, MultiLayerChebyKAN, MultiLayerTaylorKAN,GMKAN
import random
import numpy as np
import torch
from torch import optim, nn
import pandas as pd
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
import torch.nn.functional as F
import math
import torch.fft as fft
import time

def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path, header=0, dtype=dtype)
    feat = np.array(feat_df, dtype=dtype)
    return feat

def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj
def load_corr_matrix(corr_path, dtype=np.float32):
    corr_df = pd.read_csv(corr_path, header=None)
    corr = np.array(corr_df, dtype=dtype)
    return corr

def generate_dataset(
    data, seq_len, pre_len, time_len=None, normalize='max'
):
    """
    三段划分：6:2:2
    """
    if time_len is None:
        time_len = data.shape[0]
    max_val, min_val = None, None

    # 数据归一化
    if normalize == 'max':
        max_val = np.max(data)
        data = data / max_val
    elif normalize == 'zscore':
        mean_val = np.mean(data)
        std_val = np.std(data)
        data = (data - mean_val) / std_val
    elif normalize == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        data = (data - min_val) / (max_val - min_val)
    elif normalize == 'robust':
        median_val = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        data = (data - median_val) / iqr
    elif normalize == 'log':
        data = np.log1p(data)

    # 三段式划分
    train_end = int(time_len * 0.7)
    val_end = int(time_len * 0.8)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    def create_XY(data_split):
        X, Y = [], []
        for i in range(len(data_split) - seq_len - pre_len):
            X.append(data_split[i : i + seq_len])
            Y.append(data_split[i + seq_len : i + seq_len + pre_len])
        return np.array(X), np.array(Y)

    train_X, train_Y = create_XY(train_data)
    val_X, val_Y = create_XY(val_data)
    test_X, test_Y = create_XY(test_data)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y, max_val, min_val

def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, normalize=True
):
    train_X, train_Y, val_X, val_Y, test_X, test_Y, max_val, min_val = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        normalize='minmax',
    )

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_X), torch.FloatTensor(val_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, val_dataset, test_dataset, max_val, min_val

def accuracy(pred, y):
    return 1 - np.linalg.norm(y - pred) / np.linalg.norm(y)
def calculate_smape(true_values, predicted_values, epsilon=1e-8):
    """
    Calculate smoothed Mean Absolute Percentage Error (sMAPE).
    Parameters:
        true_values (numpy.array): Array of true values.
        predicted_values (numpy.array): Array of predicted values.
        epsilon (float): Small constant to avoid division by zero.
    Returns:
        float: sMAPE value.
    """
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    numerator = np.abs(predicted_values - true_values)
    denominator = np.abs(true_values) + np.abs(predicted_values)
    denominator = np.maximum(epsilon, denominator)
    smape = 2 * numerator / denominator
    smape = np.mean(smape) * 100
    return smape
def calculate_mape(true_values, predicted_values, epsilon=1e-8):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Parameters:
        true_values (numpy.array): Array of true values.
        predicted_values (numpy.array): Array of predicted values.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: MAPE value.
    """
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    mask = true_values != 0
    true_values = true_values[mask]
    predicted_values = predicted_values[mask]

    if len(true_values) == 0:
        return np.nan
    percentage_errors = np.abs((true_values - predicted_values) / true_values)
    mape = np.mean(percentage_errors) * 100
    return mape
def explained_variance(pred, y):
    return 1 - np.var(y - pred) / np.var(y)

def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian
def z_score_normalize(x):
    mean = x.mean()
    std = x.std() + 1e-8
    return (x - mean) / std
def sample_entropy(x, m=2, r=0.2):

    std_x = torch.std(x, dim=1, keepdim=True)
    r = r * std_x

    def _construct_embeddings(x, m):
        return x.unfold(1, m, 1)  # shape: (batch_size, seq_len - m + 1, m)

    emb_m = _construct_embeddings(x, m)  # shape: (batch_size, seq_len - m + 1, m)
    emb_m1 = _construct_embeddings(x, m + 1)  # shape: (batch_size, seq_len - m, m + 1)

    def _compute_pairwise_distances(embeddings):
        batch_size, n, dim = embeddings.shape
        distances = torch.cdist(embeddings, embeddings, p=2)  # shape: (batch_size, n, n)
        return distances

    dist_m = _compute_pairwise_distances(emb_m)  # shape: (batch_size, seq_len - m + 1, seq_len - m + 1)
    dist_m1 = _compute_pairwise_distances(emb_m1)  # shape: (batch_size, seq_len - m, seq_len - m)

    def _phi(distances, r):
        matches = (distances < r.unsqueeze(-1)).float().sum(dim=-1) - 1
        matches = matches / (matches.size(1) - 1)
        return matches.mean(dim=1)

    phi_m = _phi(dist_m, r)
    phi_m1 = _phi(dist_m1, r)

    sampen = -torch.log(phi_m1 / phi_m)

    return sampen
def evaluate_periodicity(x, window_size=10, step=2):
    # x  (batch_size, seq_len)
    # batch_size, seq_len = x.shape

    unfolded = x.unfold(dimension=1, size=window_size, step=step)  # (batch_size, num_windows, window_size)

    periodicity_scores = fft.fft(unfolded, dim=-1).abs().mean(dim=-1).mean(dim=-1)  # (batch_size,)
    # print('periodicity:',periodicity_scores)

    return periodicity_scores

def evaluate_smoothness(x, window_size=10, step=2):
    batch_size, seq_len = x.shape

    unfolded = x.unfold(dimension=1, size=window_size, step=step)  # (batch_size, num_windows, window_size)

    smoothness_scores = torch.diff(unfolded, dim=-1).abs().mean(dim=-1).mean(dim=-1)
    # print('smoothness_scores:',smoothness_scores)
    return smoothness_scores

def evaluate_nonlinearity(x):
    nonlinearity_scores = torch.diff(torch.diff(x, dim=1), dim=1).abs().mean(dim=-1)
    # nonlinearity_scores = sample_entropy(x, m=2, r=0.2)
    # print('nonlinearity_scores:',nonlinearity_scores)
    return nonlinearity_scores

def calculate_global_min_max(train_loader, model):
    all_periodicity_scores = []
    all_smoothness_scores = []
    all_nonlinearity_scores = []
    model.eval()

    with torch.no_grad():
        for inputs, _ in train_loader:
            inputs = inputs.to(next(model.parameters()).device)
            hidden_state = torch.zeros(inputs.size(0), model._input_dim, model._hidden_dim).to(inputs.device)

            for i in range(inputs.size(1)):
                current_input = inputs[:, i, :]
                for layer in model.tgcn_layers:
                    hidden_state = layer.graph_conv1(current_input, hidden_state)
                    hidden_state = layer.layer_norm(hidden_state)

                hidden_state_reshaped = hidden_state.view(inputs.size(0), -1)
                periodicity_score = evaluate_periodicity(hidden_state_reshaped)
                smoothness_score = evaluate_smoothness(hidden_state_reshaped)
                nonlinearity_score = evaluate_nonlinearity(hidden_state_reshaped)

                all_periodicity_scores.append(periodicity_score)
                all_smoothness_scores.append(smoothness_score)
                all_nonlinearity_scores.append(nonlinearity_score)


        global_periodicity_scores = torch.cat(all_periodicity_scores, dim=0)
        global_smoothness_scores = torch.cat(all_smoothness_scores, dim=0)
        global_nonlinearity_scores = torch.cat(all_nonlinearity_scores, dim=0)

        smallest_periodicity = torch.topk(global_periodicity_scores, k=10, largest=False).values
        largest_periodicity = torch.topk(global_periodicity_scores, k=10, largest=True).values
        smallest_smoothness = torch.topk(global_smoothness_scores, k=10, largest=False).values
        largest_smoothness = torch.topk(global_smoothness_scores, k=10, largest=True).values
        smallest_nonlinearity = torch.topk(global_nonlinearity_scores, k=10, largest=False).values
        largest_nonlinearity = torch.topk(global_nonlinearity_scores, k=10, largest=True).values

        min_periodicity = smallest_periodicity[3]
        max_periodicity = global_periodicity_scores.max().item()
        min_smoothness = smallest_smoothness[3]
        max_smoothness = global_smoothness_scores.max().item()
        min_nonlinearity = smallest_nonlinearity[3]
        max_nonlinearity = global_nonlinearity_scores.max().item()
        return {
            'min_periodicity': min_periodicity,
            'max_periodicity': max_periodicity,
            'min_smoothness': min_smoothness,
            'max_smoothness': max_smoothness,
            'min_nonlinearity': min_nonlinearity,
            'max_nonlinearity': max_nonlinearity
        }

def normalize_with_global(score, global_min, global_max):
    normalized_score = (score - global_min) / (global_max - global_min + 1e-8)
    return torch.abs(normalized_score)

def apply_low_rank_to_model(model, rank=50):
    for layer in model.tgcn_layers:
        expert_model = layer.expert_model
        # 处理 BKAN
        for kan_linear in expert_model.bspline_kan.layers:
            if not hasattr(kan_linear, 'W1_base'):
                with torch.no_grad():
                    U, S, V = torch.svd_lowrank(kan_linear.base_weight, q=rank)
                    kan_linear.W1_base = torch.nn.Parameter(U)
                    kan_linear.W2_base = torch.nn.Parameter(torch.diag(S) @ V.T)

                    flattened = kan_linear.spline_weight.reshape(
                        kan_linear.out_features, kan_linear.in_features * (kan_linear.grid_size + kan_linear.spline_order)
                    )
                    U, S, V = torch.svd_lowrank(flattened, q=rank)
                    kan_linear.W1_spline = torch.nn.Parameter(U)
                    kan_linear.W2_spline = torch.nn.Parameter(torch.diag(S) @ V.T)

                    if kan_linear.enable_standalone_scale_spline:
                        U, S, V = torch.svd_lowrank(kan_linear.spline_scaler, q=rank)
                        kan_linear.W1_scaler = torch.nn.Parameter(U)
                        kan_linear.W2_scaler = torch.nn.Parameter(torch.diag(S) @ V.T)

                    del kan_linear.base_weight
                    del kan_linear.spline_weight
                    if kan_linear.enable_standalone_scale_spline:
                        del kan_linear.spline_scaler
                    kan_linear.rank = rank
                print(f"Decomposed KANLinear: in_features={kan_linear.in_features}, out_features={kan_linear.out_features}")

        # 处理 ChebyKANLayer
        for cheby_layer in expert_model.chebyshev_kan.layers:
            if not hasattr(cheby_layer, 'W1_cheby'):
                with torch.no_grad():
                    flattened = cheby_layer.cheby_coeffs.reshape(
                        cheby_layer.input_dim, cheby_layer.out_dim * (cheby_layer.degree + 1)
                    )
                    U, S, V = torch.svd_lowrank(flattened, q=rank)
                    cheby_layer.W1_cheby = torch.nn.Parameter(U)
                    cheby_layer.W2_cheby = torch.nn.Parameter(torch.diag(S) @ V.T)

                    del cheby_layer.cheby_coeffs
                    cheby_layer.rank = rank
                print(f"Decomposed ChebyKANLayer: input_dim={cheby_layer.input_dim}, out_dim={cheby_layer.out_dim}")

        # 处理 TaylorKANLayer
        for taylor_layer in expert_model.taylor_kan.layers:
            if not hasattr(taylor_layer, 'W1_taylor'):
                with torch.no_grad():
                    flattened = taylor_layer.coeffs.reshape(
                        taylor_layer.out_dim, taylor_layer.input_dim * taylor_layer.order
                    )
                    U, S, V = torch.svd_lowrank(flattened, q=rank)
                    taylor_layer.W1_taylor = torch.nn.Parameter(U)
                    taylor_layer.W2_taylor = torch.nn.Parameter(torch.diag(S) @ V.T)

                    del taylor_layer.coeffs
                    taylor_layer.rank = rank
                print(f"Decomposed TaylorKANLayer: input_dim={taylor_layer.input_dim}, out_dim={taylor_layer.out_dim}")

adj_path = 'E:\data\\futian_adj.csv'
feat_path = 'E:\data\\futian_flow.csv'
corr_path = 'E:\data\\futian_MI.csv'

adjacency_matrix = load_adjacency_matrix(adj_path)
features = load_features(feat_path)
corr_matrix = load_corr_matrix(corr_path)
sequence_length = 12
prediction_length = 3

train_dataset, val_dataset, test_dataset, max_val, min_val = generate_torch_datasets(
    features,
    seq_len=sequence_length,
    pre_len=prediction_length,
    normalize=True
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GMKAN(adjacency_matrix, corr_matrix, hidden_dim=8, layers=1, rank=50).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model = model.to(device)
global_min_max_values = calculate_global_min_max(train_loader, model)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
criterion = torch.nn.MSELoss()

def evaluate(model, test_loader, max_val, min_val, global_min_max_values):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = targets.squeeze(1)
            outputs = model(inputs, global_min_max_values)

            outputs = outputs.view(-1) * (max_val - min_val) + min_val
            targets = targets.view(-1) * (max_val - min_val) + min_val

            true_labels.extend(targets.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    # print(predictions)
    #print(true_labels.shape)

    mse = np.mean((true_labels - predictions) ** 2)
    mae = mean_absolute_error(true_labels, predictions)
    r2 = r2_score(true_labels, predictions)
    smape_value = calculate_smape(true_labels, predictions)
    mape = calculate_mape(true_labels, predictions)
    acc = accuracy(predictions, true_labels).item()
    exp_var = explained_variance(predictions, true_labels).item()

    return true_labels, predictions, mse, mae, r2, smape_value,mape, acc, exp_var

mse_values, mae_values, r2_values, smape_values, mapes,accuracys, exp_vars = [], [], [], [], [], [],[]

for epoch in range(500):
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        targets = targets.squeeze(1)

        optimizer.zero_grad()
        outputs = model(inputs, global_min_max_values)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    val_labels, val_preds, val_mse, val_mae, val_r2, val_smape, val_mape, val_acc, val_exp = evaluate(
        model, val_loader, max_val, min_val, global_min_max_values
    )
    print(f'Val MSE: {val_mse}, MAE: {val_mae}, R2: {val_r2}, sMAPE: {val_smape}%, MAPE: {val_mape}%， ACC: {val_acc}')

    true_labels, predictions, mse, mae, r2, smape,mape, acc, exp_v = evaluate(model, test_loader, max_val, min_val, global_min_max_values)
    mse_values.append(mse)
    mae_values.append(mae)
    r2_values.append(r2)
    smape_values.append(smape)
    mapes.append(mape)
    accuracys.append(acc)
    exp_vars.append(exp_v)
    validation_loss = val_mse
    scheduler.step(validation_loss)
    model.train()

    print(f'Epoch {epoch + 1} Loss: {loss.item()}')
    print(f'MSE: {mse}, MAE: {mae}, R2 Score: {r2}, sMAPE: {smape}%, MAPE:{mape}%， accuracy: {acc},exp_var: {exp_v}')
true_labels, predictions, mse, mae, r2, sampe_value,mape,accuracy, exp_var= evaluate(model, test_loader, max_val, min_val, global_min_max_values)
print(f'test：MSE: {mse}, MAE: {mae}, R2 Score: {r2}, sMAPE: {sampe_value}%,MAPE:{mape}%，'
      f'accuracy: {accuracy},exp_var: {exp_var}')
