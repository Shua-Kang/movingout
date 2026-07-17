import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from models.gru import TrajectoryGRU
from dataset import load_data_from_huggingface, TrajectoryDataset
from moving_out.utils.states_encoding import StatesEncoder
from torch.utils.data import DataLoader

# Detect device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define the training function. The GRU is a stochastic policy: its initial
# hidden state is random noise (see models/gru.py), so sampling the same
# observation twice can produce different actions.
def train(
    model,
    dataloader,
    epochs,
    lr,
    model_name,
    device,
    action_type="fb_cos_sin",
    all_training_options=None,
):
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    min_eval_loss = None

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
            another_action,
        ) in dataloader:
            prev_states, current_state, gt_action = (
                prev_states.to(device),
                current_state.to(device),
                next_actions.to(device),
            )

            if action_type == "cos_sin":
                normalizaed_action = torch.cat(
                    [
                        (gt_action[:, :, 0] * torch.cos(gt_action[:, :, 1])).unsqueeze(2),
                        (gt_action[:, :, 0] * torch.sin(gt_action[:, :, 1])).unsqueeze(2),
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

            steps += 1
            optimizer.zero_grad()

            # Forward pass
            mse_output, ce_logits, _ = model(prev_states, current_state)

            # Compute loss
            if action_type == "cos_sin":
                loss_mse = criterion_mse(mse_output, gt_action[:, :, :2])
            elif action_type == "fb_cos_sin":
                loss_mse = criterion_mse(mse_output[:, :, 0:1], gt_action[:, :, 0:1])
                loss_mse += criterion_mse(mse_output[:, :, 1:3], normalizaed_action)

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
        current_time = datetime.now()
        print(
            current_time.strftime("%Y-%m-%d %H:%M:%S")
            + f"_Epoch [{epoch+1}/{epochs}], Loss: {total_loss/steps:.4f}, total_loss_ce: {total_loss_ce/steps}, total_loss_mse: {total_loss_mse/steps}"
        )

        eval_loss = avg_train_loss
        if min_eval_loss is None:
            min_eval_loss = eval_loss + 0.1
        if avg_train_loss < min_eval_loss:
            model_dict = {"training_options": all_training_options, "model": model}
            torch.save(model_dict, model_name)
            min_eval_loss = eval_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a stochastic GRU behavior-cloning agent (random-noise "
        "initial hidden state). One model per robot, each on its own data only."
    )
    parser.add_argument(
        "--dataset_name", type=str, default="ShuaKang/movingout_task1", help="dataset name in huggingface"
    )
    parser.add_argument("--split", default="train", type=str, help="dataset split")
    parser.add_argument(
        "--map_name", default=None, type=str,
        help="only train on trajectories from this map (e.g. HandOff); default: all maps",
    )
    parser.add_argument(
        "--robot_id", default="both", type=str, choices=["1", "2", "both"],
        help="which robot's model to train; each robot uses only its own data",
    )
    parser.add_argument("--model_save_path", default="gru.pt", type=str, help="model path; _robot1/_robot2 is appended")
    parser.add_argument("--action_type", default="fb_cos_sin", type=str)
    parser.add_argument("--obs_horizon", default=5, type=int)
    parser.add_argument("--action_horizon", default=3, type=int)
    parser.add_argument("--epoch", default=1000, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--hidden_dim", default=1024, type=int)
    parser.add_argument("--drop_out", action="store_true")

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    obs_horizon = args.obs_horizon
    action_horizon = args.action_horizon

    robot_ids = [1, 2] if args.robot_id == "both" else [int(args.robot_id)]
    states_encoder = StatesEncoder()
    for robot_id in robot_ids:
        loaded_data = load_data_from_huggingface(
            args.dataset_name,
            split=args.split,
            states_encoder=states_encoder,
            map_name=args.map_name,
            robot_id=robot_id,
        )
        dataset = TrajectoryDataset(loaded_data, obs_horizon, action_horizon, states_encoder=states_encoder)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        one_state_dim = len(loaded_data[0][0][0])

        # Initialize the model (GRU consumes the state sequence directly)
        model = TrajectoryGRU(
            state_dim=one_state_dim,
            hidden_dim=args.hidden_dim,
            m=obs_horizon,
            k=action_horizon,
            action_type=args.action_type,
            dropout=args.drop_out,
        )
        all_training_options = vars(args).copy()
        all_training_options["arch"] = "gru"
        all_training_options["robot_id"] = robot_id
        all_training_options["previous_steps"] = obs_horizon
        all_training_options["selected_actions"] = action_horizon

        save_path = args.model_save_path.replace(".pt", f"_robot{robot_id}.pt")
        print(f"=== training GRU for robot_{robot_id} ({len(dataset)} samples) -> {save_path}")
        train(
            model,
            dataloader,
            epochs=args.epoch,
            lr=args.lr,
            model_name=save_path,
            device=device,
            action_type=args.action_type,
            all_training_options=all_training_options,
        )
