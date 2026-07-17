from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch.utils.data import DataLoader

from models.diffusion_policy import ConditionalUnet1D
from dataset import load_data_from_huggingface, TrajectoryDataset
from moving_out.utils.states_encoding import StatesEncoder

# Detect device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Diffusion action layout per step: [fb, cos(angle), sin(angle), hold_onehot x2]
ACTION_DIM = 5


def encode_actions(next_actions):
    """[B, k, 3] (fb, angle, hold01) -> [B, k, 5] diffusion targets."""
    fb = next_actions[:, :, 0:1]
    cos = torch.cos(next_actions[:, :, 1:2])
    sin = torch.sin(next_actions[:, :, 1:2])
    onehot_hold = torch.nn.functional.one_hot(
        next_actions[:, :, 2].long(), num_classes=2
    ).float()
    return torch.cat([fb, cos, sin, onehot_hold], dim=-1)


def get_diffusion_loss(model, obs_batch, action_batch, timesteps, noise, noise_scheduler):
    noisy_actions = noise_scheduler.add_noise(action_batch, noise, timesteps)
    noise_pred = model(noisy_actions, timesteps, obs_batch)
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(action_batch, noise, timesteps)
    elif noise_scheduler.config.prediction_type == "sample":
        target = action_batch
    else:
        raise TypeError("prediction type not recognized.")
    loss = nn.functional.mse_loss(noise_pred, target)

    return loss


def train_model_step(model, obs, action, optimizer, scheduler, noise_scheduler):
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (action.shape[0],), device=device
    ).long()
    noise = torch.empty(action.shape, device=device).normal_(-1, 1)

    loss = get_diffusion_loss(model, obs, action, timesteps, noise, noise_scheduler)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss


# Define the training function. The diffusion policy is stochastic: actions are
# denoised from random noise at inference time.
def train(model, dataloader, epochs, lr, model_name, device, num_train_timesteps=32,
          all_training_options=None):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler(
        optimizer=optimizer,
        name="cosine",
        num_warmup_steps=500,
        num_training_steps=epochs * len(dataloader),
    )
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    model.to(device)
    min_total_loss = None
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        for prev_states, current_state, next_actions, another_action in dataloader:
            steps += 1
            prev_states = prev_states.to(device)
            current_state = current_state.to(device)
            next_actions = next_actions.to(device)

            # Condition on the flattened observation window
            obs = torch.cat(
                [prev_states.reshape(prev_states.size(0), -1), current_state], dim=1
            )
            action_targets = encode_actions(next_actions)

            loss = train_model_step(
                model, obs, action_targets, optimizer, scheduler, noise_scheduler
            )
            total_loss += loss.item()
        if min_total_loss is None:
            min_total_loss = total_loss + 0.1
        if min_total_loss > total_loss:
            model_dict = {"training_options": all_training_options, "model": model}
            torch.save(model_dict, model_name)
            min_total_loss = total_loss
        current_time = datetime.now()
        print(
            current_time.strftime("%Y-%m-%d %H:%M:%S")
            + f"_Epoch [{epoch+1}/{epochs}], Loss: {total_loss/steps:.4f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a stochastic diffusion-policy behavior-cloning agent. "
        "One model per robot, each on its own robot's data only."
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
    parser.add_argument("--model_save_path", default="dp.pt", type=str, help="model path; _robot1/_robot2 is appended")
    parser.add_argument("--obs_horizon", default=1, type=int)
    parser.add_argument(
        "--action_horizon", default=8, type=int,
        help="diffusion prediction horizon (actions denoised per inference)",
    )
    parser.add_argument(
        "--selected_actions", default=2, type=int,
        help="how many of the predicted actions are executed at evaluation time",
    )
    parser.add_argument("--epoch", default=1000, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_train_timesteps", default=32, type=int)

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
        obs_dim = obs_horizon * one_state_dim + one_state_dim

        model = ConditionalUnet1D(
            input_dim=ACTION_DIM,
            use_obs_encoder=True,
            global_cond_dim=obs_dim,
            diffusion_step_embed_dim=128,
            down_dims=[128, 256, 512],
            kernel_size=5,
            n_groups=8,
        )

        all_training_options = vars(args).copy()
        all_training_options["arch"] = "dp"
        all_training_options["robot_id"] = robot_id
        all_training_options["previous_steps"] = obs_horizon
        all_training_options["pred_horizon"] = action_horizon
        all_training_options["action_dim"] = ACTION_DIM

        save_path = args.model_save_path.replace(".pt", f"_robot{robot_id}.pt")
        print(f"=== training diffusion policy for robot_{robot_id} ({len(dataset)} samples) -> {save_path}")
        train(
            model,
            dataloader,
            epochs=args.epoch,
            lr=args.lr,
            model_name=save_path,
            device=device,
            num_train_timesteps=args.num_train_timesteps,
            all_training_options=all_training_options,
        )
