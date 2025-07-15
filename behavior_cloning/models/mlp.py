import torch
import torch.nn as nn

# Detect device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define the MLP model
class TrajectoryMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        k,
        action_type="fb_cos_sin",
        Probabilistic=False,
        dropout=False,
    ):
        super(TrajectoryMLP, self).__init__()
        self.Probabilistic = Probabilistic

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.Tanh()
        if dropout:
            self.dropout = nn.Dropout(0.1)
        else:
            self.dropout = False
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.fc3 = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4))

        if self.Probabilistic:
            output_size = 4 * k
        else:
            if action_type == "fb_cos_sin":
                self.output_mse_size = 3
            elif action_type == "cos_sin":
                self.output_mse_size = 2
            elif action_type == "ce_cos_sin":
                self.output_mse_size = 5
            elif action_type == "ce_cos_sin_speed_direction":
                self.output_mse_size = 5
            output_size = self.output_mse_size * k
        self.fc3_mse = nn.Linear(
            int(hidden_dim / 4), output_size
        )  # MSE output layer for k steps, 2 outputs each
        self.fc3_ce = nn.Linear(
            int(hidden_dim / 4), 2 * k
        )  # CrossEntropy output layer for k steps, 2 outputs each

    def forward(self, prev_states, current_state, return_inner_state=False):
        # Concatenate previous states and current state
        x = torch.cat(
            (prev_states.reshape(prev_states.size(0), -1), current_state), dim=1
        )
        x = self.fc1(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        if return_inner_state:
            inner_state = x
        else:
            inner_state = None
        x = self.fc3(x)
        x = self.relu(x)
        # Separate MSE and CE outputs
        mse_output = self.fc3_mse(x).reshape(
            x.size(0), -1, self.output_mse_size
        )  # Output shape [batch_size, k, 2]
        ce_logits = self.fc3_ce(x).reshape(
            x.size(0), -1, 2
        )  # Output shape [batch_size, k, 2]
        if self.Probabilistic:
            mean, log_std = mse_output.chunk(2, dim=1)
            std = torch.exp(log_std)
            # print(mse_output.shape)
            # print(mean.shape)
            # print(log_variance.shape)
            # exit(0)
            return mean, std, ce_logits
        else:
            return mse_output, ce_logits, inner_state


class TrajectoryMLPEncodeAnother(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, k, another_state_dim=3, fusion_method="add"
    ):
        super(TrajectoryMLPEncodeAnother, self).__init__()

        self.fusion_method = fusion_method

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)
        self.fc2_another = nn.Linear(another_state_dim, hidden_dim)
        if self.fusion_method == "cat":
            self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        elif self.fusion_method == "add" or self.fusion_method == "mul":
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        else:
            print("ERROR : ")
            exit(0)
        self.fc3 = nn.Linear(hidden_dim, int(hidden_dim / 2))

        self.fc3_mse = nn.Linear(
            int(hidden_dim / 2), 2 * k
        )  # MSE output layer for k steps, 2 outputs each
        self.fc3_ce = nn.Linear(int(hidden_dim / 2), 2 * k)

        # self.fc_predict_action_mse = nn.Linear(
        #     hidden_dim, 2 * k
        # )  # MSE output layer for k steps, 2 outputs each
        # self.fc_predict_action_ce = nn.Linear(hidden_dim, 2 * k)

    def forward(self, prev_states, current_state, another_state):
        # Concatenate previous states and current state
        x = torch.cat(
            (prev_states.reshape(prev_states.size(0), -1), current_state), dim=1
        )
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x2 = self.fc2_another(another_state)
        x2 = self.relu(x2)
        x2 = self.dropout_2(x2)
        if self.fusion_method == "add":
            x = x + x2
        elif self.fusion_method == "mul":
            x = x * x2
        elif self.fusion_method == "cat":
            x = torch.cat((x, x2), dim=1)
        else:
            print(f"ERROR: {self.fusion_method}")
            exit(0)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)

        # another_encoding = torch.cat((another_mse_output.squeeze(1), another_ce_logits.squeeze(1)), dim=1)
        # x = torch.cat((x, another_encoding), dim=1)
        # x = self.fc3(x)
        # x = self.relu(x)
        # Separate MSE and CE outputs
        mse_output = self.fc3_mse(x).reshape(
            x.size(0), -1, 2
        )  # Output shape [batch_size, k, 2]
        ce_logits = self.fc3_ce(x).reshape(
            x.size(0), -1, 2
        )  # Output shape [batch_size, k, 2]

        return mse_output, ce_logits


class TrajectoryMLPPrecitAnother(nn.Module):
    def __init__(self, input_dim, hidden_dim, k, another_state_dim=3):
        super(TrajectoryMLPPrecitAnother, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim + another_state_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, int(hidden_dim / 2))

        self.fc3_mse = nn.Linear(
            int(hidden_dim / 2), 2 * k
        )  # MSE output layer for k steps, 2 outputs each
        self.fc3_ce = nn.Linear(int(hidden_dim / 2), 2 * k)

        # self.fc_predict_action_mse = nn.Linear(
        #     hidden_dim, 2 * k
        # )  # MSE output layer for k steps, 2 outputs each
        # self.fc_predict_action_ce = nn.Linear(hidden_dim, 2 * k)

    def forward(self, prev_states, current_state, another_state):
        # Concatenate previous states and current state
        x = torch.cat(
            (prev_states.reshape(prev_states.size(0), -1), current_state), dim=1
        )
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = torch.cat((x, another_state), dim=1)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)

        # another_encoding = torch.cat((another_mse_output.squeeze(1), another_ce_logits.squeeze(1)), dim=1)
        # x = torch.cat((x, another_encoding), dim=1)
        # x = self.fc3(x)
        # x = self.relu(x)
        # Separate MSE and CE outputs
        mse_output = self.fc3_mse(x).reshape(
            x.size(0), -1, 2
        )  # Output shape [batch_size, k, 2]
        ce_logits = self.fc3_ce(x).reshape(
            x.size(0), -1, 2
        )  # Output shape [batch_size, k, 2]

        return mse_output, ce_logits


if __name__ == "__main__":
    input_dim = 170 * 3
    hidden_dim = 1024
    k = 2

    model = TrajectoryMLPPrecitAnother(input_dim, hidden_dim, k)

    prvious_states = torch.zeros(4, 2, 170)
    current_states = torch.zeros(4, 170)
    another = torch.zeros(4, 3)

    output = model(prvious_states, current_states, another)
    print(output)


class RankingModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(RankingModel, self).__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
        )
        self.score_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state, actions):
        # Encode state
        state_feat = self.state_encoder(state)  # Shape: [batch_size, hidden_dim]

        # Encode each action sequence
        batch_size = actions.shape[0]
        horizon = actions.shape[2]  # Assuming actions shape: [batch_size, 4, 4, 3]
        actions = actions.view(
            -1, horizon, actions.size(-1)
        )  # Flatten batch and candidates
        action_feat = self.action_encoder(actions).mean(dim=1)  # Average over horizon

        # Combine state and action features
        combined_feat = state_feat.repeat_interleave(4, dim=0) + action_feat
        scores = self.score_layer(combined_feat).view(batch_size, 4)  # Reshape back
        return scores
