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


def save_q_network(q_network, file_path):
    torch.save(q_network.state_dict(), file_path)


def load_q_network(file_path, input_dimension, hidden_dimension=512):
    q_network = QNetwork(input_dimension, hidden_dimension)
    state_dictionary = torch.load(file_path, map_location=torch.device("cpu"))
    q_network.load_state_dict(state_dictionary)
    q_network.eval()
    return q_network
