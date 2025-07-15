import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.mlp import TrajectoryMLP
from dataset import load_data_from_huggingface, TrajectoryDataset
from moving_out.utils.states_encoding import StatesEncoder
from torch.utils.data import DataLoader
# Detect device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from datetime import datetime

# Define the training function
def train(
    model,
    dataloader,
    epochs,
    lr,
    model_name,
    device,
    predict_another,
    action_type="fb_cos_sin",
    all_training_options=None,
):
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)  # Move model to the selected device
    min_eval_loss = None

    train_losses = []
    eval_losses = []

    for epoch in range(epochs):
        total_loss = 0
        total_loss_ce = 0
        total_loss_mse = 0
        total_items = 0
        steps = 0
        for (
            prev_states,
            current_state,
            next_actions,
            another_action
        ) in dataloader:
            prev_states, current_state, next_actions, another_action = (
                prev_states.to(device),
                current_state.to(device),
                next_actions.to(device),
                another_action.to(device),
            )
            if predict_another:
                gt_action = another_action
            else:
                gt_action = next_actions
                gt_action[:, :, 0] = gt_action[:, :, 0]

                if action_type == "cos_sin":
                    normalizaed_action = torch.cat(
                        [
                            (
                                gt_action[:, :, 0] * torch.cos(gt_action[:, :, 1])
                            ).unsqueeze(2),
                            (
                                gt_action[:, :, 0] * torch.sin(gt_action[:, :, 1])
                            ).unsqueeze(2),
                        ],
                        dim=2,
                    )
                    gt_action[:, :, 0:2] = normalizaed_action
                elif action_type == "fb_cos_sin":
                    normalizaed_action = torch.cat(
                        [
                            (torch.cos(gt_action[:, :, 1])).unsqueeze(2),
                            (torch.sin(gt_action[:, :, 1])).unsqueeze(2),
                        ],
                        dim=2,
                    )
                    # print(normalizaed_action.shape)
                elif action_type == "ce_cos_sin":

                    def normalize_action(action):
                        normalized_action = torch.zeros_like(action, dtype=torch.int)
                        normalized_action[action > 0.5] = 1
                        normalized_action[action < -0.5] = 2
                        return normalized_action

                    gt_action[:, :, 0] = normalize_action(gt_action[:, :, 0])
                    normalizaed_action = torch.cat(
                        [
                            (torch.cos(gt_action[:, :, 1])).unsqueeze(2),
                            (torch.sin(gt_action[:, :, 1])).unsqueeze(2),
                        ],
                        dim=2,
                    )
                elif action_type == "ce_cos_sin_speed_direction":

                    def normalize_action(action):
                        normalized_action = torch.zeros_like(action, dtype=torch.int)
                        normalized_action[action > 0.5] = 1
                        normalized_action[action < -0.5] = 2
                        return normalized_action

                    gt_action[:, :, 0] = normalize_action(gt_action[:, :, 0])

                    speed_direction = torch.where(
                        gt_action[:, :, 0] == 2, torch.tensor(-1), gt_action[:, :, 0]
                    )
                    speed_direction = torch.where(
                        speed_direction == 0, torch.tensor(1), speed_direction
                    )

                    normalizaed_action = torch.cat(
                        [
                            (torch.cos(gt_action[:, :, 1]) * speed_direction).unsqueeze(
                                2
                            ),
                            (torch.sin(gt_action[:, :, 1]) * speed_direction).unsqueeze(
                                2
                            ),
                        ],
                        dim=2,
                    )

            steps += 1
            prev_states, current_state, gt_action = (
                prev_states.to(device),
                current_state.to(device),
                gt_action.to(device),
            )
            optimizer.zero_grad()

            # Forward pass
            mse_output, ce_logits, _ = model(prev_states, current_state)
            # print("mse_output: , gt_action", mse_output[0], gt_action[0, :, :2])
            # Compute loss
            if action_type == "cos_sin":
                loss_mse = criterion_mse(mse_output, gt_action[:, :, :2])
            elif action_type == "fb_cos_sin":
                loss_mse = criterion_mse(mse_output[:, :, 0:1], gt_action[:, :, 0:1])
                loss_mse += criterion_mse(mse_output[:, :, 1:3], normalizaed_action)
            elif action_type == "ce_cos_sin":
                loss_mse = criterion_ce(
                    mse_output[:, :, 0:3].reshape(-1, 3),
                    gt_action[:, :, 0].reshape(-1).long(),
                )
                loss_mse += criterion_mse(mse_output[:, :, 3:5], normalizaed_action)
            elif action_type == "ce_cos_sin_speed_direction":
                loss_mse = criterion_ce(
                    mse_output[:, :, 0:3].reshape(-1, 3),
                    gt_action[:, :, 0].reshape(-1).long(),
                )
                loss_mse += criterion_mse(mse_output[:, :, 3:5], normalizaed_action)

            loss_ce = criterion_ce(
                ce_logits.reshape(-1, 2), gt_action[:, :, 2].reshape(-1).long()
            )
            loss = loss_mse + loss_ce

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_ce += loss_ce.item()
            total_loss_mse += loss_mse.item()
            total_items += prev_states.shape[0]
        avg_train_loss = total_loss / total_items
        train_losses.append(avg_train_loss)
        current_time = datetime.now()
        print(
            current_time.strftime("%Y-%m-%d %H:%M:%S")
            + f"_Epoch [{epoch+1}/{epochs}], Loss: {total_loss/steps:.4f}, total_loss_ce: {total_loss_ce/steps}, total_loss_mse: {total_loss_mse/steps}"
        )

        eval_loss = avg_train_loss
        if min_eval_loss is None:
            min_eval_loss = eval_loss + 0.1
        if avg_train_loss < min_eval_loss:
            if all_training_options is not None:
                model_dict = {"training_options": all_training_options, "model": model}
            else:
                model_dict = {"training_options": None, "model": model}
            torch.save(model_dict, model_name)
            min_eval_loss = eval_loss

    


if __name__ == "__main__":
    # Configuration parameters
    import argparse

    parser = argparse.ArgumentParser(description="Process JSON file and ID.")
    parser.add_argument(
        "--dataset_name", type=str, default="ShuaKang/movingout_task1", help="dataset name in huggingface"
    )
    parser.add_argument("--model_save_path", default="model.pt", type=str, help="model path")
    parser.add_argument(
        "--action_type", default="fb_cos_sin", type=str, help="model path"
    )
    parser.add_argument("--obs_horizon", default=5, type=int, help="model path")
    parser.add_argument("--action_horizon", default=3, type=int, help="model path")
    parser.add_argument("--epoch", default=1000, type=int, help="model path")
    parser.add_argument("--predict_another", action="store_true")
    parser.add_argument("--drop_out", action="store_true")
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--shift_another_states", action="store_true")
    parser.add_argument("--no_another_states", action="store_true")
    parser.add_argument("--noise_std", default=0.1, type=float, help="model path")
    parser.add_argument("--shift_range", default=1, type=int, help="model path")
    parser.add_argument("--recombination_trajectories", action="store_true")

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    
    obs_horizon = args.obs_horizon  # Previous steps
    action_horizon = args.action_horizon  # Next steps
    
    states_encoder = StatesEncoder()
    loaded_data = load_data_from_huggingface(args.dataset_name, split="all", states_encoder=states_encoder)
    dataset = TrajectoryDataset(loaded_data, obs_horizon, action_horizon, states_encoder=states_encoder)
    batch_size = 1024
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    one_state_dim = len(loaded_data[0][0][0])
    
    input_dim = (
        obs_horizon * one_state_dim + one_state_dim
    ) 
    hidden_dim = 2048

    # Initialize the model
    model = TrajectoryMLP(
        input_dim, hidden_dim, action_horizon, action_type=args.action_type, dropout=args.drop_out
    )
    all_training_options = vars(args)
    # Train the model
    train(
        model,
        dataloader,
        epochs=args.epoch,
        lr=0.001,
        model_name=args.model_save_path,
        device=device,
        predict_another=args.predict_another,
        action_type=args.action_type,
        all_training_options=all_training_options,
    )
