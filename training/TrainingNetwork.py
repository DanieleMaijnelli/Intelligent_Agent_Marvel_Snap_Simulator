import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_dimension, hidden_dimension=512):
        super(QNetwork, self).__init__()
        self.linear_layer_1 = nn.Linear(input_dimension, hidden_dimension)
        self.linear_layer_2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.linear_layer_3 = nn.Linear(hidden_dimension, 1)

    def forward(self, state_action_tensor):
        x = torch.relu(self.linear_layer_1(state_action_tensor))
        x = torch.relu(self.linear_layer_2(x))
        x = self.linear_layer_3(x)
        return x.squeeze(-1)


class QNetworkA(nn.Module):
    def __init__(self, input_dimension, bottleneck_dim=512, mid_dim=2048):
        super(QNetworkA, self).__init__()

        self.linear_layer_1 = nn.Linear(input_dimension, bottleneck_dim)
        self.layer_norm_1 = nn.LayerNorm(bottleneck_dim)

        self.linear_layer_2 = nn.Linear(bottleneck_dim, mid_dim)
        self.layer_norm_2 = nn.LayerNorm(mid_dim)

        self.linear_layer_3 = nn.Linear(mid_dim, bottleneck_dim)
        self.layer_norm_3 = nn.LayerNorm(bottleneck_dim)

        self.linear_layer_4 = nn.Linear(bottleneck_dim, 1)

    def forward(self, state_action_tensor):
        x = self.linear_layer_1(state_action_tensor)
        x = self.layer_norm_1(x)
        x = torch.relu(x)

        x = self.linear_layer_2(x)
        x = self.layer_norm_2(x)
        x = torch.relu(x)

        x = self.linear_layer_3(x)
        x = self.layer_norm_3(x)
        x = torch.relu(x)

        x = self.linear_layer_4(x)
        return x.squeeze(-1)


class QNetworkB(nn.Module):
    def __init__(self, input_dimension, hidden_dim=512, num_hidden_layers=4):
        super(QNetworkB, self).__init__()

        assert num_hidden_layers >= 1

        self.input_layer = nn.Linear(input_dimension, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_norms.append(nn.LayerNorm(hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state_action_tensor):
        x = self.input_layer(state_action_tensor)
        x = self.input_norm(x)
        x = torch.relu(x)

        for layer, norm in zip(self.hidden_layers, self.hidden_norms):
            x = layer(x)
            x = norm(x)
            x = torch.relu(x)

        x = self.output_layer(x)
        return x.squeeze(-1)


def save_q_network(q_network, file_path):
    torch.save(q_network.state_dict(), file_path)


def load_q_network(file_path, input_dimension, architecture="base", hidden_dimension=512,
                   bottleneck_dim=512, mid_dim=2048, num_hidden_layers=4):
    if architecture == "base":
        q_network = QNetwork(input_dimension, hidden_dimension)
    elif architecture == "A":
        q_network = QNetworkA(input_dimension, bottleneck_dim=bottleneck_dim, mid_dim=mid_dim)
    elif architecture == "B":
        q_network = QNetworkB(input_dimension, hidden_dim=hidden_dimension, num_hidden_layers=num_hidden_layers, )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    state_dictionary = torch.load(file_path, map_location=torch.device("cpu"))
    q_network.load_state_dict(state_dictionary)
    q_network.eval()
    return q_network
