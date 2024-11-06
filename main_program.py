from model import BKAN, MultiLayerChebyKAN, MultiLayerTaylorKAN
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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

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
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize='max'
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
        max_val, min_val = None, None
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
        quartile1 = np.percentile(data, 25)
        quartile3 = np.percentile(data, 75)
        iqr = quartile3 - quartile1
        data = (data - median_val) / iqr
    elif normalize == 'log':
        data = np.log1p(data)

    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i : i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i : i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y), max_val, min_val

def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y, max_val, min_val= generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize='minmax',
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset, max_val, min_val

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
        # print("Smallest 10 Periodicity Scores:", smallest_periodicity)
        # print("Largest 10 Periodicity Scores:", largest_periodicity)
        # print("Smallest 10 Smoothness Scores:", smallest_smoothness)
        # print("Largest 10 Smoothness Scores:", largest_smoothness)
        # print("Smallest 10 Nonlinearity Scores:", smallest_nonlinearity)
        # print("Largest 10 Nonlinearity Scores:", largest_nonlinearity)
        # print("Global Periodicity Scores:", global_periodicity_scores)
        # print("Global Smoothness Scores:", global_smoothness_scores)
        # print("Global Nonlinearity Scores:", global_nonlinearity_scores)

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


class ExpertModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ExpertModel, self).__init__()
        self.chebyshev_kan = MultiLayerChebyKAN([input_dim, output_dim], degree=3)  # Chebyshev KAN
        self.bspline_kan = BKAN([input_dim, output_dim])
        self.taylor_kan = MultiLayerTaylorKAN([input_dim, output_dim], order=3)

    def forward(self, x, periodicity_scores, smoothness_scores, nonlinearity_scores):
        batch_size = x.size(0)
        # print(periodicity_scores)
        gates = torch.stack([periodicity_scores,smoothness_scores,nonlinearity_scores], dim=1)
        gates = torch.abs(gates)
        # gates_sum = gates.sum(dim=1, keepdim=True)
        # gates = gates / (gates_sum + 1e-8)

        gates = F.softmax(gates, dim=1)

        chebyshev_input = gates[:, 0].view(-1, 1) * x
        # print(gates[:, 0].view(-1, 1) )
        bspline_input = gates[:, 1].view(-1, 1) * x
        taylor_input = gates[:, 2].view(-1, 1) * x

        chebyshev_output = self.chebyshev_kan(chebyshev_input)
        bspline_output = self.bspline_kan(bspline_input)
        taylor_output = self.taylor_kan(taylor_input)

        output = chebyshev_output + bspline_output + taylor_output
        # print(output)
        return output


class GraphConvolution(nn.Module):
    def __init__(self, adj, corr, input_dim: int, output_dim: int, bias: float = 0.0):
        super(GraphConvolution, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self.register_buffer("corr", calculate_laplacian_with_self_loop(torch.FloatTensor(corr)))
        self.weights = nn.Parameter(
            torch.FloatTensor(self._input_dim + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        node_feature_mean = torch.mean(inputs, dim=0)
        alpha = torch.sigmoid(node_feature_mean)
        combined_matrix = alpha * self.laplacian + (1 - alpha) * self.corr
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, _input_dim + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, input_dim + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (input_dim + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self.input_dim) * batch_size)
        )
        # A[x, h] (num_nodes, (input_dim + 1) * batch_size)
        a_times_concat = combined_matrix @ concatenation
        # A[x, h] (num_nodes, input_dim + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._input_dim + 1, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, input_dim + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, input_dim + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._input_dim + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        # outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }

class GMKANCEll(nn.Module):
    def __init__(self, adj, corr,input_dim: int, hidden_dim: int, num_nodes):
        super(GMKANCEll, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.corr = corr
        self.graph_conv1 = GraphConvolution(
            self.adj, self.corr, self._hidden_dim, self._hidden_dim , bias=1.0
        )
        self.layer_norm = nn.LayerNorm(self._hidden_dim)
        self.expert_model = ExpertModel(num_nodes * hidden_dim, num_nodes * hidden_dim)

    def forward(self, inputs, hidden_state, global_min_max_values):
        # print("Initial hidden_state shape:", hidden_state.shape)
        hidden_state = self.graph_conv1(inputs, hidden_state)
        hidden_state = self.layer_norm(hidden_state)
        # print("After graph_conv1 shape:", hidden_state.shape)
        batch_size, num_nodes, hidden_dim = hidden_state.shape
        hidden_state_reshaped = hidden_state.reshape(batch_size, num_nodes * hidden_dim)
        # print("Before KANLayer shape:", hidden_state_reshaped.shape)

        periodicity_scores = evaluate_periodicity(hidden_state_reshaped, window_size=10, step=2)  # (batch_size,)
        # print("Periodicity Scores in GKANCell:", periodicity_scores)
        # 2.
        smoothness_scores = evaluate_smoothness(hidden_state_reshaped, window_size=10, step=2)  # (batch_size,)
        # print("Smoothness Scores in GKANCell:", smoothness_scores)
        # 3.
        nonlinearity_scores = evaluate_nonlinearity(hidden_state_reshaped).to(inputs.device)  # (batch_size,)
        # print("Nonlinearity Scores in GKANCell:", nonlinearity_scores)

        periodicity_scores = normalize_with_global(periodicity_scores, global_min_max_values['min_periodicity'],
                                                   global_min_max_values['max_periodicity'])
        smoothness_scores = normalize_with_global(smoothness_scores, global_min_max_values['min_smoothness'],
                                                  global_min_max_values['max_smoothness'])
        nonlinearity_scores = normalize_with_global(nonlinearity_scores, global_min_max_values['min_nonlinearity'],
                                                    global_min_max_values['max_nonlinearity'])
        # print(periodicity_scores,smoothness_scores,nonlinearity_scores)
        smoothness_scores = 1-smoothness_scores

        new_hidden_state = self.expert_model(hidden_state_reshaped, periodicity_scores, smoothness_scores,
                                             nonlinearity_scores)

        #  hidden_state (batch_size, num_nodes, hidden_dim)
        new_hidden_state = new_hidden_state.reshape(batch_size, num_nodes, hidden_dim)
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}

class GMKAN(nn.Module):
    def __init__(self, adj, corr,  hidden_dim: int, layers:int=1,  **kwargs):
        super(GMKAN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.layers = layers
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.corr = torch.FloatTensor(corr)
        # self.corr= torch.FloatTensor(corr)
        self.tgcn_layers = nn.ModuleList([
            GMKANCEll(self.adj, self.corr,self._input_dim if i == 0 else self._hidden_dim, self._hidden_dim, self._input_dim)
            for i in range(layers)
        ])
        self.final_layer = nn.Linear(self._hidden_dim, 1)

    def forward(self, inputs, global_min_max_values, prediction_steps:int=3, training=True, current_epoch=1, total_epochs=200):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes, self._hidden_dim).type_as(
            inputs
        )
        initial_teacher_forcing_ratio = 1.0 
        final_teacher_forcing_ratio = 0.0 
        teacher_forcing_ratio = initial_teacher_forcing_ratio - (current_epoch / total_epochs) * (
                    initial_teacher_forcing_ratio - final_teacher_forcing_ratio)
        outputs = []
        for i in range(seq_len):
            for layer in self.tgcn_layers:
                output, hidden_state = layer(inputs[:, i, :], hidden_state, global_min_max_values)
        current_input = inputs[:,-1,:]
        for step in range(prediction_steps):
            for layer in self.tgcn_layers:
                output, hidden_state = layer(current_input, hidden_state, global_min_max_values)
            output = hidden_state.view(batch_size * num_nodes, self._hidden_dim)
            output = self.final_layer(output)
            output = output.view(batch_size, num_nodes)
            outputs.append(output.unsqueeze(1))
            current_input = output
            if training and random.random() < teacher_forcing_ratio:
                current_input = inputs[:, step, :]  
            else:
                current_input = output
        outputs = torch.cat(outputs, dim=1)
        return outputs

adj_path = 'E:\data\\futian_adj.csv'
feat_path = 'E:\data\\futian_flow.csv'
corr_path = 'E:\data\\futian_MI.csv'

adjacency_matrix = load_adjacency_matrix(adj_path)
features = load_features(feat_path)
corr_matrix = load_corr_matrix(corr_path)
sequence_length = 12
prediction_length = 3

train_dataset, test_dataset, max_val, min_val = generate_torch_datasets(
    features,
    seq_len=sequence_length,
    pre_len=prediction_length,
    normalize=True
)
# print("Max Value:", max_val, "Min Value:", min_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = GMKAN(adjacency_matrix, corr_matrix, hidden_dim=8)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model = model.to(device)

global_min_max_values = calculate_global_min_max(train_loader, model)

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
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

for epoch in range(200):
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        targets = targets.squeeze(1)

        optimizer.zero_grad()
        outputs = model(inputs, global_min_max_values)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    true_labels, predictions, mse, mae, r2, smape,mape, acc, exp_v = evaluate(model, test_loader, max_val, min_val, global_min_max_values)
    mse_values.append(mse)
    mae_values.append(mae)
    r2_values.append(r2)
    smape_values.append(smape)
    mapes.append(mape)
    accuracys.append(acc)
    exp_vars.append(exp_v)
    validation_loss = mse
    scheduler.step(validation_loss)
    model.train()

    print(f'Epoch {epoch + 1} Loss: {loss.item()}')
    print(f'MSE: {mse}, MAE: {mae}, R2 Score: {r2}, sMAPE: {smape}%, MAPE:{mape}%， accuracy: {acc},exp_var: {exp_v}')

true_labels, predictions, mse, mae, r2, sampe_value,mape,accuracy, exp_var= evaluate(model, test_loader, max_val, min_val, global_min_max_values)
print(f'test：MSE: {mse}, MAE: {mae}, R2 Score: {r2}, sMAPE: {sampe_value}%,MAPE:{mape}%，'
      f'accuracy: {accuracy},exp_var: {exp_var}')

