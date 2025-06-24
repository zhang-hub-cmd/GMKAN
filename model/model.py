import torch
from torch import nn
import torch.nn.functional as F
import math
from main_program import *

class LowRankLayer(nn.Module):
    def __init__(self, in_features, out_features, rank, expanded_dim=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.expanded_dim = expanded_dim or in_features

        self.W1 = nn.Parameter(torch.Tensor(out_features, rank))  # [输出, 秩]
        self.W2 = nn.Parameter(torch.Tensor(rank, self.expanded_dim))  # [秩, 扩展输入]
        # 缩放参数
        self.scale1 = nn.Parameter(torch.ones(out_features))
        self.scale2 = nn.Parameter(torch.ones(rank))
        # 初始化权重
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

    def forward(self, x):
        W1_scaled = self.W1 * self.scale1.unsqueeze(1)  
        W2_scaled = self.W2 * self.scale2.unsqueeze(1) 
        intermediate = F.linear(x, W2_scaled) 
        output = F.linear(intermediate, W1_scaled)

        return output


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            rank=None,
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.rank = rank
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.degree = grid_size + spline_order

        # 初始化样条控制点
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        if rank is None:
            self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
            self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, self.degree))
            if enable_standalone_scale_spline:
                self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        else:
            self.base_low_rank = LowRankLayer(in_features, out_features, rank)
            self.spline_low_rank = LowRankLayer(in_features, out_features, rank, in_features * self.degree)

        self.reset_parameters()

    def reset_parameters(self):
        if self.rank is None:
            torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
            with torch.no_grad():
                noise = (
                        (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2)
                        * self.scale_noise
                        / self.grid_size
                )
                self.spline_weight.data.copy_(
                    (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                    * self.curve2coeff(
                        self.grid.T[self.spline_order: -self.spline_order],
                        noise,
                    )
                )
                if self.enable_standalone_scale_spline:
                    torch.nn.init.kaiming_uniform_(
                        self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                    )

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        if self.rank is None:
            base_output = F.linear(self.base_activation(x), self.base_weight)
            spline_basis = self.b_splines(x).view(x.size(0), -1)
            spline_output = F.linear(
                spline_basis,
                self.scaled_spline_weight.view(self.out_features, -1),
            )
        else:
            # 使用低秩分解
            base_output = self.base_low_rank(self.base_activation(x))

            # 样条部分
            spline_basis = self.b_splines(x).view(x.size(0), -1)
            spline_output = self.spline_low_rank(spline_basis)

        return base_output + spline_output
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        self.grid.copy_(grid.T)
        if self.rank is None:
            self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))
        else:
            result = self.curve2coeff(x, unreduced_spline_output)
            flattened = result.reshape(self.out_features, self.in_features * self.degree)
            U, S, V = torch.svd_lowrank(flattened, q=self.rank)
            self.W1_spline.data.copy_(U)
            self.W2_spline.data.copy_(torch.diag(S) @ V.T)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        if self.rank is None:
            l1_fake = self.spline_weight.abs().mean(-1)
            regularization_loss_activation = l1_fake.sum()
            p = l1_fake / regularization_loss_activation
            regularization_loss_entropy = -torch.sum(p * p.log())
        else:
            l1_fake = (self.W1_spline @ self.W2_spline).abs().mean(-1)
            regularization_loss_activation = l1_fake.sum()
            p = l1_fake / regularization_loss_activation
            regularization_loss_entropy = -torch.sum(p * p.log())
            if self.enable_standalone_scale_spline:
                l1_scaler = (self.W1_scaler @ self.W2_scaler).abs().mean(-1)
                regularization_loss_scaler = l1_scaler.sum()
                p_scaler = l1_scaler / regularization_loss_scaler
                entropy_scaler = -torch.sum(p_scaler * p_scaler.log())
            else:
                regularization_loss_scaler = 0.0
                entropy_scaler = 0.0
            regularization_loss_activation += regularization_loss_scaler
            regularization_loss_entropy += entropy_scaler
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class BKAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            rank=None,
    ):
        super(BKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    rank=rank,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class TaylorKANLayer(nn.Module):
    def __init__(self, input_dim, out_dim, order, addbias=True, rank=None):
        super(TaylorKANLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.order = order
        self.addbias = addbias
        self.rank = rank
        self.expanded_dim = input_dim * order

        if rank is None:
            self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, order) * 0.01)
        else:
            self.low_rank = LowRankLayer(self.expanded_dim, out_dim, rank)

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x):
        shape = x.shape
        x = x.view(-1, self.input_dim)

        x_powers = torch.cat([x ** i for i in range(self.order)], dim=1)

        if self.rank is None:
            x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1)
            coeffs = self.coeffs
            y = torch.zeros((x.shape[0], self.out_dim), device=x.device)
            for i in range(self.order):
                term = (x_expanded ** i) * coeffs[:, :, i]
                y += term.sum(dim=-1)
        else:
            y = self.low_rank(x_powers)

        if self.addbias:
            y += self.bias

        return y.view(shape[0], -1, self.out_dim) if len(shape) > 2 else y


class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, rank=None):
        super(ChebyKANLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.degree = degree
        self.rank = rank
        self.expanded_dim = input_dim * (degree + 1)

        if rank is None:
            self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
            nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        else:
            # 使用通用低秩层
            self.low_rank = LowRankLayer(self.expanded_dim, output_dim, rank)

        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # 基函数计算
        x = torch.tanh(x)
        x = x.view((-1, self.input_dim, 1)).expand(-1, -1, self.degree + 1)
        x = x.acos()
        x *= self.arange
        x = x.cos()  # [B, input_dim, degree+1]

        # 展平输入
        x_flat = x.view(-1, self.expanded_dim)

        if self.rank is None:
            y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs)
        else:
            y = self.low_rank(x_flat)

        return y.view(-1, self.out_dim)


class MultiLayerChebyKAN(nn.Module):
    def __init__(self, layers_hidden, degree, rank=None):
        super(MultiLayerChebyKAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(ChebyKANLayer(in_features, out_features, degree, rank=rank))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MultiLayerTaylorKAN(nn.Module):
    def __init__(self, layers_hidden, order, addbias=True, rank=None):
        super(MultiLayerTaylorKAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(TaylorKANLayer(in_features, out_features, order, addbias, rank=rank))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ExpertModel(nn.Module):
    def __init__(self, input_dim, output_dim, rank=50):
        super(ExpertModel, self).__init__()
        self.chebyshev_kan = MultiLayerChebyKAN([input_dim, output_dim], degree=3, rank=rank)
        self.bspline_kan = BKAN([input_dim, output_dim], rank=rank)
        self.taylor_kan = MultiLayerTaylorKAN([input_dim, output_dim], order=3, rank=rank)

    def forward(self, x, periodicity_scores, smoothness_scores, nonlinearity_scores):
        batch_size = x.size(0)
        gates = torch.stack([periodicity_scores, smoothness_scores, nonlinearity_scores], dim=1)
        gates = torch.abs(gates)
        gates = F.softmax(gates, dim=1)

        chebyshev_input = gates[:, 0].view(-1, 1) * x
        bspline_input = gates[:, 1].view(-1, 1) * x
        taylor_input = gates[:, 2].view(-1, 1) * x

        chebyshev_output = self.chebyshev_kan(chebyshev_input)
        bspline_output = self.bspline_kan(bspline_input)
        taylor_output = self.taylor_kan(taylor_input)

        output = chebyshev_output + bspline_output + taylor_output
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
        node_mean = torch.mean(inputs, dim=0)
        # 动态计算基准阈值（增强自适应）
        global_mean = torch.mean(node_mean)  # 系统整体流量水平
        mu_0 = 0.3 + 0.4 * torch.sigmoid(global_mean)  # 阈值在[0.3,0.7]间动态调整
        # 非线性权重函数
        k = 6.0  # 控制非线性陡峭程度
        alpha = 1 / (1 + torch.exp(k * (node_mean - mu_0)))
        combined_matrix = alpha * self.laplacian + (1 - alpha) * self.corr
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._output_dim)
        )
        # [x, h] (batch_size, num_nodes, _input_dim + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, input_dim + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (input_dim + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._input_dim+1) * batch_size)
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
