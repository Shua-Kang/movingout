import torch
import torch.nn as nn
import torch.optim as optim


class TrajectoryGRU(nn.Module):
    def __init__(
        self,
        state_dim, 
        hidden_dim,
        m, 
        k, 
        action_type="fb_cos_sin",
        Probabilistic=False,
        dropout=False,
    ):
        """
        Unlike the original MLP, here we need to pass in:
        - state_dim: dimension of a single state vector
        - m: how many past frames
        - k: how many future action frames to predict
        Other logic like hidden_dim, action_type etc. can be reused
        """
        super(TrajectoryGRU, self).__init__()
        self.Probabilistic = Probabilistic
        self.action_type = action_type
        self.m = m
        self.k = k
        self.hidden_dim = hidden_dim
        # Define GRU, input dimension is state_dim, hidden_dim can be set as needed
        self.gru = nn.GRU(
            input_size=state_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True
        )

        # For dropout, it can be used after fully connected layers or GRU output
        self.do_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(0.1)
        else:
            self.dropout = None

        # Several fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.relu = nn.Tanh()  # You can change to ReLU according to needs

        # Determine output size based on action_type
        if self.Probabilistic:
            # If it's a probabilistic model, e.g., mean+std, size 4*k (2*k for mean, 2*k for std)
            output_size = 4 * k
        else:
            # Otherwise decide according to original logic
            if action_type == "fb_cos_sin":
                self.output_mse_size = 3
            elif action_type == "cos_sin":
                self.output_mse_size = 2
            elif action_type == "ce_cos_sin":
                self.output_mse_size = 5
            elif action_type == "ce_cos_sin_speed_direction":
                self.output_mse_size = 5
            output_size = self.output_mse_size * k

        # Used to predict MSE part (regression part, e.g., cos,sin)
        self.fc3_mse = nn.Linear(hidden_dim // 4, output_size)

        # Used to predict CE part (classification part, e.g., forward/backward etc.)
        self.fc3_ce = nn.Linear(hidden_dim // 4, 2 * k)

    def forward(self, prev_states, current_state, return_inner_state=False):
        """
        Parameters:
        - prev_states: [batch_size, m, state_dim]
        - current_state: [batch_size, state_dim]
        - return_inner_state: whether to return intermediate features

        Returns:
        - mse_output: [batch_size, k, output_mse_size]  (if Probabilistic=False)
        - ce_logits:  [batch_size, k, 2]
        - inner_state: optional intermediate features
        or
        - mean, std, ce_logits (if Probabilistic=True)
        """

        # First concatenate (prev_states, current_state) into sequence: [batch_size, m+1, state_dim]
        current_state = current_state.unsqueeze(1)  # [batch_size, 1, state_dim]
        x = torch.cat([prev_states, current_state], dim=1)

        # Feed into GRU
        # out: [batch_size, m+1, hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim] here is single layer, so hidden.shape = [1, batch_size, hidden_dim]
        batch_size = x.size(0)
        h0 = torch.randn(1, batch_size, self.hidden_dim, device=x.device)
        out, hidden = self.gru(x, h0)

        # We take the hidden state of the last time step (or directly use hidden[-1])
        # hidden.shape = [1, batch_size, hidden_dim] -> becomes [batch_size, hidden_dim]
        h_last = hidden[0]

        # Connect several fully connected layers
        x = self.fc1(h_last)
        x = self.relu(x)
        if self.do_dropout:
            x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)

        if return_inner_state:
            inner_state = x
        else:
            inner_state = None

        # Final fully connected layer
        x_for_mse = self.fc3_mse(x)  # [batch_size, output_size]
        x_for_ce = self.fc3_ce(x)  # [batch_size, 2*k]

        # reshape
        # MSE corresponding output: [batch_size, k, output_mse_size]
        mse_output = x_for_mse.reshape(x.size(0), -1, self.output_mse_size)

        # CE corresponding output: [batch_size, k, 2]
        ce_logits = x_for_ce.reshape(x.size(0), -1, 2)

        if self.Probabilistic:
            # If it's probabilistic output
            mean, log_std = mse_output.chunk(2, dim=1)
            std = torch.exp(log_std)
            return mean, std, ce_logits
        else:
            return mse_output, ce_logits, inner_state


# Define the GRU model
# class TrajectoryGRU(nn.Module):
#     def __init__(self, input_dim, hidden_dim, k, num_layers=1):
#         super(TrajectoryGRU, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc_mse = nn.Linear(hidden_dim, 2 * k)  # MSE output layer for k steps, 2 outputs each
#         self.fc_ce = nn.Linear(hidden_dim, 2 * k)  # CrossEntropy output layer for k steps, 2 outputs each

#     def forward(self, prev_states, current_state):
#         # Concatenate previous states and current state
#         x = torch.cat((prev_states, current_state.unsqueeze(1)), dim=1)
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # Initialize hidden state

#         # GRU forward pass
#         out, _ = self.gru(x, h0)

#         # Use the last time step's output for predictions
#         out = out[:, -1, :]

#         # Separate MSE and CE outputs
#         mse_output = self.fc_mse(out).reshape(out.size(0), -1, 2)  # Output shape [batch_size, k, 2]
#         ce_logits = self.fc_ce(out).reshape(out.size(0), -1, 2)  # Output shape [batch_size, k, 2]

#         return mse_output, ce_logits
