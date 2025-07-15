import copy
import random

import numpy as np
import torch
from moving_out.benchmarks.moving_out import MovingOutEnv
from moving_out.utils.states_encoding import StatesEncoder
from torch.utils.data import DataLoader, Dataset


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        all_trajectories,
        obs_horizon,
        action_horizon,
        add_noise=False,
        noise_std=0.1,
        recombination_trajectories=False,
        recombination_analyzer=None,
        img_obs=False,
        img_obs_map_id=None,
        states_encoder=None,
    ):
        self.img_obs = img_obs
        self.pair_data = all_trajectories
        self.data = []
        for data in self.pair_data:
            self.data += [data]
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.recombination_trajectories = recombination_trajectories

        if self.recombination_trajectories:
            from moving_out.utils.discretizer import Discretizer
            from moving_out.utils.trajectory_path_analyzer import TrajectoryPathAnalyzer
            grid_size = [-23, 24]
            self.discretizer = Discretizer(-1.2, 1.2, grid_size[0], grid_size[1])
            self.recombination_analyzer = TrajectoryPathAnalyzer(grid_size = grid_size)
        if(states_encoder is None):
            self.states_encoder = StatesEncoder()
        else:
            self.states_encoder = states_encoder
        

    def __len__(self):
        return sum(len(traj) for traj in self.data)

    def get_item_by_traj_idx(
        self, traj_idx, index, img_obs=False, recomb_idx_index=None
    ):
        # calculate the item indx then use self.get_item_by_indx
        traj = self.data[traj_idx]
        # Current step
        state = traj[index][0]

        repected_beginning = 0
        repected_end = 0

        # Previous obs_horizon steps (with padding)
        prev_states = []
        for i in range(self.obs_horizon):
            if index - i - 1 >= 0:
                prev_states.append(traj[index - i - 1][0])
            else:
                prev_states.append(traj[0][0])  # padding
                repected_beginning += 1
        # Reverse to get chronological order
        prev_states.reverse()

        

        if img_obs:
            traj_json = self.json_data[traj_idx]
            selected_traj = traj_json[index][0]
            prev_json_states = []
            for i in range(self.obs_horizon):
                if index - i - 1 >= 0:
                    prev_json_states.append(traj_json[index - i - 1][0])
                else:
                    prev_json_states.append(traj_json[0][0])  # padding
                    # repected_beginning += 1
            prev_json_states.reverse()

        # Next action_horizon steps with n actions starting from t (with padding)
        next_actions = []
        the_other_agent_actions = []
        for i in range(self.action_horizon):
            if index + i < len(traj):
                action = traj[index + i][1][0]
                the_other_agent_action = traj[index + i][1][1]
                next_actions.append(action)
                the_other_agent_actions.append(the_other_agent_action)
            else:
                next_actions.append(traj[-1][1][0])
                the_other_agent_actions.append(traj[-1][1][1])  # padding
                repected_end += 1
        # Convert to tensors
        prev_states = torch.tensor(prev_states, dtype=torch.float32)
        next_actions = torch.tensor(next_actions, dtype=torch.float32)
        the_other_agent_action = torch.tensor(
            the_other_agent_actions, dtype=torch.float32
        )
        state = torch.tensor(state, dtype=torch.float32)
        if self.add_noise:
            noise_std = self.noise_std
            noise_mean = 0

            if not img_obs:
                # self.robot_1_obs_index = self.states_encoder.get_robot_1_obs_index()
                self.robot_2_obs_index = self.states_encoder.get_robot_2_obs_index()
                noise = torch.randn_like(state[self.robot_2_obs_index[0]:self.robot_2_obs_index[1]]) * noise_std + noise_mean
                state[self.robot_2_obs_index[0]:self.robot_2_obs_index[1]] += noise
                noise = torch.randn_like(prev_states[:, self.robot_2_obs_index[0]:self.robot_2_obs_index[1]]) * noise_std + noise_mean
                prev_states[:, self.robot_2_obs_index[0]:self.robot_2_obs_index[1]] += noise
            else:
                if recomb_idx_index is not None:
                    traj_idx, traj_index = recomb_idx_index

                    recomb_traj_json = self.json_data[traj_idx]
                    recomb_selected_traj = recomb_traj_json[index][0]
                    selected_traj["states"]["robot_2"]["pos"] = recomb_selected_traj[
                        "states"
                    ]["robot_2"]["pos"]
                    selected_traj["states"]["robot_2"]["angle"] = recomb_selected_traj[
                        "states"
                    ]["robot_2"]["angle"]

                    recomb_prev_json_states = []
                    for i in range(self.obs_horizon):
                        if index - i - 1 >= 0:
                            recomb_prev_json_states.append(
                                recomb_traj_json[index - i - 1][0]
                            )
                        else:
                            recomb_prev_json_states.append(recomb_traj_json[0][0])
                    for pev_states, recomb_pev_states in zip(
                        prev_json_states, recomb_prev_json_states
                    ):
                        selected_traj = copy.deepcopy(selected_traj)
                        selected_traj["states"]["robot_2"]["pos"] = recomb_pev_states[
                            "states"
                        ]["robot_2"]["pos"]
                        selected_traj["states"]["robot_2"]["angle"] = recomb_pev_states[
                            "states"
                        ]["robot_2"]["angle"]

                selected_traj = copy.deepcopy(selected_traj)
                selected_traj["states"]["robot_2"]["pos"] = np.array(
                    selected_traj["states"]["robot_2"]["pos"]
                ) + (np.random.randn(2) * noise_std + noise_mean)
                selected_traj["states"]["robot_2"]["angle"] = np.array(
                    selected_traj["states"]["robot_2"]["angle"]
                ) + (np.random.randn(1) * noise_std + noise_mean)
                self.env.update_env_by_given_state(selected_traj)
                current_img_obs = self.env.render("rgb_array")
                previous_img_obs = []
                for pev_states in prev_json_states:
                    selected_traj = copy.deepcopy(pev_states)
                    selected_traj["states"]["robot_2"]["pos"] = np.array(
                        selected_traj["states"]["robot_2"]["pos"]
                    ) + (np.random.randn(2) * noise_std + noise_mean)
                    selected_traj["states"]["robot_2"]["angle"] = np.array(
                        selected_traj["states"]["robot_2"]["angle"]
                    ) + (np.random.randn(1) * noise_std + noise_mean)
                    self.env.update_env_by_given_state(selected_traj)
                    robot_1_obs = self.env.render("rgb_array")
                    # current_img_obs = robot_1_obs
                    previous_img_obs.append(robot_1_obs)

        else:
            if img_obs:
                self.env.update_env_by_given_state(selected_traj)
                current_img_obs = self.env.render("rgb_array")
                previous_img_obs = []
                for pev_states in prev_json_states:
                    self.env.update_env_by_given_state(pev_states)
                    robot_1_obs = self.env.render("rgb_array")
                    previous_img_obs.append(robot_1_obs)

        if not img_obs:
            previous_img_obs = "_"
            current_img_obs = "_"

        return (
            prev_states,
            state,
            next_actions,
            the_other_agent_action,
            repected_beginning,
            repected_end,
            previous_img_obs,
            current_img_obs,
        )

    def get_item_by_indx(self, index, img_obs=False, recomb_idx_index=None):
        traj_idx, traj_index = self._locate_traj_by_indx(index)
        (
            prev_states,
            state,
            next_actions,
            the_other_agent_action,
            repected_beginning,
            repected_end,
            previous_img_obs,
            current_img_obs,
        ) = self.get_item_by_traj_idx(
            traj_idx, traj_index, img_obs=img_obs, recomb_idx_index=recomb_idx_index
        )
        return (
            prev_states,
            state,
            next_actions,
            the_other_agent_action,
            repected_beginning,
            repected_end,
            previous_img_obs,
            current_img_obs,
        )

    def _locate_traj_by_indx(self, index):
        traj_idx = 0
        ori_index = index
        while index >= len(self.data[traj_idx]):
            index -= len(self.data[traj_idx])
            traj_idx += 1

        return traj_idx, index

    def __getitem__(self, index):
        # Find the corresponding trajectory
        if self.img_obs:
            if self.env is None:
                self.env = MovingOutEnv(map_name=self.img_obs_map_id)
        if index >= len(self):
            print(f"Index {index} is out of bounds!")
        else:
            pass

        if self.recombination_trajectories:
            (
                prev_states,
                state,
                next_actions,
                the_other_agent_action,
                repected_beginning,
                repected_end,
                previous_img_obs,
                current_img_obs,
            ) = self.get_item_by_indx(index, self.img_obs)

            if repected_end != 0:
                pass
            else:
                end_index = index + self.action_horizon - 1
                (
                    end_prev_states,
                    end_state,
                    end_next_actions,
                    end_the_other_agent_action,
                    repected_beginning,
                    repected_end,
                    _,
                    _,
                ) = self.get_item_by_indx(end_index, img_obs=False)
                start_pos = self.discretizer.discretize(
                    [float(state[0]), float(state[1])]
                )
                end_pos = self.discretizer.discretize(
                    [float(end_state[0]), float(end_state[1])]
                )
                result = self.recombination_analyzer.query_by_start_and_end(
                    start_pos, end_pos
                )
                if len(result["trajectories"]) > 1:
                    traj_idx, _, start_indx, end_indx = random.choice(
                        result["trajectories"]
                    )
                    if not self.img_obs:
                        (
                            recomb_prev_states,
                            recomb_state,
                            recomb_next_actions,
                            recomb_the_other_agent_action,
                            _,
                            _,
                            recomb_previous_img_obs,
                            recomb_current_img_obs,
                        ) = self.get_item_by_traj_idx(traj_idx, start_indx)
                        state[8:16] = recomb_state[8:16]
                        prev_states[:, 8:16] = recomb_prev_states[:, 8:16]
                    else:
                        (
                            recomb_prev_states,
                            recomb_state,
                            recomb_next_actions,
                            recomb_the_other_agent_action,
                            _,
                            _,
                            recomb_previous_img_obs,
                            recomb_current_img_obs,
                        ) = self.get_item_by_traj_idx(traj_idx, start_indx)

        else:
            (
                prev_states,
                state,
                next_actions,
                the_other_agent_action,
                _,
                _,
                previous_img_obs,
                current_img_obs,
            ) = self.get_item_by_indx(index, img_obs=self.img_obs)
        if(self.img_obs):
            return (
                prev_states,
                state,
                next_actions,
                the_other_agent_action,
                previous_img_obs,
                current_img_obs,
            )
        else:
            return (
                prev_states,
                state,
                next_actions,
                the_other_agent_action,
            )

    


def load_data_from_huggingface(dataset_name, split="all", states_encoder=None):
    if(states_encoder is None):
        states_encoder = StatesEncoder()
    def relabel_actions(actions, current_robot_states, next_robot_states):
        if actions[2] and (current_robot_states["hold"] != next_robot_states["hold"]):
            return 1
        else:
            return 0
    from datasets import load_dataset
    import json
    ds = load_dataset(dataset_name, split=split)
    
    all_trajectories = []
    
    for data in ds:
        data_dict = json.loads(data["steps_data"])
        trajectory_robot_1 = []
        trajectory_robot_2 = []
        for i, data_item in enumerate(data_dict):
            states_action_pairs_0 = []
            states_action_pairs_1 = []
            action = data_item[2]
            if i != len(data_dict) - 1:
                action[0][2] = relabel_actions(
                    action[0],
                    data_dict[i][1]["robot_1"],
                    data_dict[i + 1][1]["robot_1"],
                )
                action[1][2] = relabel_actions(
                    action[1],
                    data_dict[i][1]["robot_2"],
                    data_dict[i + 1][1]["robot_2"],
                )
            else:
                action[0][2] = 1 if action[0][2] else 0
                action[1][2] = 1 if action[1][2] else 0
        
            encoded_states = states_encoder.get_state_by_current_obs_states(data_item[1])
            states_action_pairs_0.append((encoded_states[0], (action[0], action[1])))
            states_action_pairs_1.append((encoded_states[1], (action[1], action[0])))
        all_trajectories.append(states_action_pairs_0)
        all_trajectories.append(states_action_pairs_1)
    return all_trajectories


if __name__ == "__main__":
    dataset_name = "ShuaKang/movingout_task1"
    states_encoder = StatesEncoder()
    loaded_data = load_data_from_huggingface(dataset_name, split="all", states_encoder=states_encoder)
    obs_horizon = 5  # past observation
    action_horizon = 3  # future actions

    dataset = TrajectoryDataset(loaded_data, obs_horizon, action_horizon, states_encoder=states_encoder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("Dataset size: len: ", len(dataset))
    for prev_states, state, next_actions, the_other_agent_action in dataloader:
        print("Previous States: ", prev_states.shape)
        print("Current State: ", state.shape)
        print("Next Actions: ", next_actions.shape)
        print("The other agents Actions: ", the_other_agent_action.shape)
        break
