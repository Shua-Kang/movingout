"""A user interface for teleoperating an agent in an x-magical environment.

Modified from https://github.com/unixpickle/obs-tower2/blob/master/obs_tower2/recorder/env_interactor.py
"""

import copy
import json
import os
import random
import time
from datetime import datetime

import cv2
import imageio
import numpy as np
import torch
from moving_out.benchmarks.moving_out import MovingOutEnv
from moving_out.utils.states_encoding import StatesEncoder
from moving_out.utils.utils import (append_step_to_file,
                                    calculate_average_results,
                                    init_trajectory_file, states_buffer)


class MovingOutEvaluator:
    """User interface for interacting in an x-magical environment."""

    def __init__(
        self,
        action_dim=0,
        map_name=1000,
        max_item_number=7,
        replan_times=1,
        model_1_replan_times=1,
        model_2_replan_times=1,
        add_noise_to_item=None,
        experiment_save_path=None,
        save_replan_data=None,
        ranking_model=None,
        replan_strategy=None,
        another_action_predictor=None,
    ):
        self.max_item_number = max_item_number
        self._action_dim = action_dim
        self.replan_times = replan_times
        self.model_1_replan_times = model_1_replan_times
        self.model_2_replan_times = model_2_replan_times
        self.use_2_action_dim = True
        self.experiment_save_path = experiment_save_path
        self.hold = [False, False]
        self.axis = [[0, 0, 0, 0], [0, 0, 0, 0]]

        self.env = MovingOutEnv(use_state=False, map_name=map_name)
        if self.env.if_cache_founded:
            self.cahce_file = copy.deepcopy(self.env.distance_cache)
        self.buffer_robot_1 = None
        self.buffer_robot_2 = None
        self.reset_buffer()
        self.add_noise_to_item = add_noise_to_item
        self.states_convetor = StatesEncoder(max_number_shaps=self.max_item_number)

        self.save_replan_data = save_replan_data
        self.ranking_model = ranking_model
        self.replan_strategy = replan_strategy
        self.another_action_predictor = another_action_predictor

    def reset_buffer(self, length=1):
        self.buffer_robot_1 = states_buffer(length)
        self.buffer_robot_2 = states_buffer(length)

    def reset_buffer(self, robot_1_previous_step=1, robot_2_previous_step=1):
        self.buffer_robot_1 = states_buffer(robot_1_previous_step)
        self.buffer_robot_2 = states_buffer(robot_2_previous_step)

    def get_action(
        self,
        robot_1_states,
        buffer_robot_1,
        robot_2_states,
        buffer_robot_2,
        model_1,
        model_2,
    ):
        action_1_displaces, action_1_holdings = model_1.get_action(
            robot_1_states, buffer_robot_1.get_states()
        )

        action_1s = []

        for action_1_displace, action_1_holding in zip(
            action_1_displaces, action_1_holdings
        ):
            action_1_holding = True if action_1_holding[0].item() == 1 else False
            # action_1_holding = True

            action_1 = [
                action_1_displace[0].item(),
                action_1_displace[1].item(),
                action_1_holding,
            ]
            action_1s.append(action_1)

        action_2_displaces, action_2_holdings = model_2.get_action(
            robot_2_states, buffer_robot_2.get_states()
        )
        action_2s = []
        for action_2_displace, action_2_holding in zip(
            action_2_displaces, action_2_holdings
        ):
            action_2_holding = True if action_2_holding[0].item() == 1 else False
            action_2 = [
                action_2_displace[0].item(),
                action_2_displace[1].item(),
                action_2_holding,
            ]
            action_2s.append(action_2)

        actions = []
        for a1, a2 in zip(action_1s, action_2s):
            actions.append([a1, a2])
        # action = [action_1, action_2]
        return actions

    def evaluate_ids(
        self,
        ids,
        model_1,
        model_2,
        evaluate_times=1,
        max_steps=300,
        save_videos=True,
        model_horizon=1,
        file_name=["", ""],
    ):
        evaluation_results = {"model_1": file_name[0], "model_2": file_name[1]}
        evaluation_results["training_options_1"] = model_1.training_options
        evaluation_results["training_options_2"] = model_2.training_options
        evaluation_results["evaluation_options"] = {
            "replan_times": self.replan_times,
            "replan_strategy": self.replan_strategy,
            "add_noise_to_item": self.add_noise_to_item,
        }

        self.model_1_previous_steps = evaluation_results["training_options_1"][
            "previous_steps"
        ]
        self.model_2_previous_steps = evaluation_results["training_options_2"][
            "previous_steps"
        ]

        if save_videos:
            while True:
                time.sleep(random.random())
                current_time = time.time()
                milliseconds = int((current_time - int(current_time)) * 1000)
                formatted_time = (
                    time.strftime("%Y-%m-%d!%H:%M:%S", time.localtime())
                    .replace("-", "_")
                    .replace(" ", "_")
                    .replace(":", "_")
                )
                formatted_time = formatted_time.replace("!", "/")
                formatted_time_with_ms = f"{formatted_time}_{milliseconds:03d}"
                dir_path = formatted_time_with_ms
                dir_path = os.path.join(self.experiment_save_path, dir_path)
                dir_path = os.path.join("exp_results", dir_path)

                if os.path.exists(dir_path):
                    time.sleep(random.random())
                    continue
                else:
                    try:
                        os.makedirs(dir_path)
                    except:
                        continue
                    break
        for id in ids:
            self.env.reset(map_name=id, add_noise_to_item=self.add_noise_to_item)
            self.reset_buffer(self.model_1_previous_steps, self.model_2_previous_steps)
            evaluation_results[str(id)] = {}
            evaluation_results["map_name"] = [str(id)]

            for et in range(evaluate_times):
                # if(self.replan_strategy == "vae_next_state_predictor"):
                self.predcitor_recorder = []
                start_time = datetime.now()
                last_time = time.time()
                steps = 0
                images_list = []
                saved_actions_so_far = []
                if self.save_replan_data is not None:
                    save_action_selection_data = []
                self.reset_buffer(
                    self.model_1_previous_steps, self.model_2_previous_steps
                )
                self.env.reset(map_name=id, add_noise_to_item=self.add_noise_to_item)
                steps_data = []
                json_file_path = init_trajectory_file(id, dir_path)
                for j in range(max_steps):
                    all_states = self.env.get_all_states()
                    encoded_states = (
                        self.states_convetor.get_state_by_current_obs_states(all_states)
                    )
                    robot_1_states = encoded_states[0]
                    robot_2_states = encoded_states[1]
                    self.buffer_robot_1.push_states(robot_1_states)
                    self.buffer_robot_2.push_states(robot_2_states)

                    actions_scores = {}
                    for rp_t in range(self.replan_times):
                        actions = self.get_action(
                            robot_1_states=robot_1_states,
                            buffer_robot_1=self.buffer_robot_1,
                            robot_2_states=robot_2_states,
                            buffer_robot_2=self.buffer_robot_2,
                            model_1=model_1,
                            model_2=model_2,
                        )

                        # self.cloned_env = copy.deepcopy(self.env)
                        if self.replan_times != 1:
                            # img = self.env.render("rgb_array")
                            # cv2.imwrite(f"img_1.png", img)
                            if self.env.if_cache_founded:
                                self.cloned_env = self.env.clone(self.cahce_file)
                            else:
                                self.cloned_env = self.env.clone()
                            # img = self.cloned_env.render("rgb_array")
                            # cv2.imwrite(f"img_2.png", img)
                        else:
                            self.cloned_env = self.env
                        images_list_in_cloned_env = []
                        states_in_cloned_env = []
                        steps_cloned_env = steps
                        current_state = self.cloned_env.get_all_states()
                        data_for_save_trajectory = []
                        for a in actions:
                            steps_cloned_env += 1
                            obs, rew, done, info = self.cloned_env.step(a)
                            img_obs = self.cloned_env.render("rgb_array")
                            states = self.cloned_env.get_all_states()
                            states_in_cloned_env.append(states)
                            images_list_in_cloned_env.append(img_obs)
                            global_score = self.cloned_env.global_score()
                            global_dense_score = self.cloned_env.global_dense_reward()
                            data_for_save_trajectory.append(
                                [steps_cloned_env, states, rew, done, a]
                            )
                            if global_score >= 0.999 or done or steps >= max_steps:
                                padding_image = len(actions) - len(
                                    images_list_in_cloned_env
                                )
                                for _ in range(padding_image):
                                    images_list_in_cloned_env.append(img_obs)
                                break

                        if self.replan_strategy == "clone_env":
                            predicted_dense_reward = global_dense_score
                        elif self.replan_strategy == "clone_env_with_another_predictor":
                            pass
                        elif self.replan_strategy == "vae_next_state_predictor":
                            current_tensor_0 = torch.tensor(
                                self.states_convetor.get_state_by_current_obs_states(
                                    current_state
                                )[0]
                            )
                            current_tensor_1 = torch.tensor(
                                self.states_convetor.get_state_by_current_obs_states(
                                    current_state
                                )[1]
                            )
                            actions = actions
                            agent_0_action = []
                            agent_1_action = []
                            for action in actions:
                                agent_0_action.append(action[0])
                                agent_1_action.append(action[1])
                            agent_0_action = torch.tensor(
                                np.array(agent_0_action, dtype=float)
                            )
                            agent_1_action = torch.tensor(
                                np.array(agent_1_action, dtype=float)
                            )
                            score_0 = self.ranking_model.predict(
                                current_tensor,
                                agent_0_action,
                                agent_1_action,
                                self.env,
                                self.states_convetor,
                            )
                            # score_1 = self.ranking_model.predict(
                            #     current_tensor, agent_1_action, self.env, self.states_convetor
                            # )
                            predicted_dense_reward = score_0

                            # _, _ = self.ranking_model.predict(current_tensor, agent_0_action)
                            # the_best_one = actions_scores[the_best_one_index]
                        else:
                            predicted_dense_reward = global_dense_score
                        # exit(0)
                        actions_scores[rp_t] = {
                            "actions": actions,
                            "images_list": images_list_in_cloned_env,
                            "global_score": global_dense_score,
                            "predicted_dense_reward": predicted_dense_reward,
                            "steps": steps_cloned_env,
                            "done": done,
                            "env": self.cloned_env,
                            "after_action_states": states_in_cloned_env,
                            "state": current_state,
                            "data_for_save_trajectory": data_for_save_trajectory,
                        }

                    def calculate_pos_difference_of_two_robots(state_1, state_2):
                        robot_0_pos_diff = sum(
                            np.abs(
                                np.array(state_1["robot_1"]["pos"])
                                - np.array(state_2["robot_1"]["pos"])
                            )
                        )
                        robot_1_pos_diff = sum(
                            np.abs(
                                np.array(state_1["robot_2"]["pos"])
                                - np.array(state_2["robot_2"]["pos"])
                            )
                        )
                        return robot_0_pos_diff, robot_1_pos_diff

                    def check_if_the_same(actions_scores):
                        global_score = actions_scores[0]["global_score"]
                        equal = True
                        for v in actions_scores.values():
                            if global_score == v:
                                pass
                            else:
                                return actions_scores
                        if equal:
                            for k, v in zip(
                                actions_scores.keys(), actions_scores.values()
                            ):
                                _max_distance = calculate_pos_difference_of_two_robots(
                                    v[0]["state"], v[-1]["state"]
                                )
                                actions_scores[k]["global_score"] = float(
                                    sum(_max_distance)
                                )
                            return actions_scores
                        else:
                            return actions_scores

                    def check_if_all_actions_are_the_same(actions_scores):
                        global_score = actions_scores[0]["global_score"]
                        equal = True
                        for v in actions_scores.values():
                            if global_score != v["global_score"]:
                                return False
                        return True

                    if self.replan_times > 1 and self.replan_strategy == "clone_env":
                        actions_scores = check_if_the_same(actions_scores)
                    # if(self.replan_times > 1 and self.replan_strategy == "vae_next_state_predictor"):
                    #     if(not actions_scores[0]["state"]["robot_1"]["hold"]  and  not actions_scores[0]["state"]["robot_2"]["hold"]):
                    #         return actions_scores
                    elif (
                        self.replan_times > 1
                        and self.replan_strategy == "vae_next_state_predictor"
                        and not check_if_all_actions_are_the_same(actions_scores)
                    ):

                        def check_if_predicted_the_same_as_gt(actions_scores):
                            max_key_based_on_prediction = max(
                                actions_scores,
                                key=lambda k: actions_scores[k][
                                    "predicted_dense_reward"
                                ],
                            )

                            max_key_based_on_gt = max(
                                actions_scores,
                                key=lambda k: actions_scores[k]["global_score"],
                            )
                            return max_key_based_on_prediction == max_key_based_on_gt

                        self.predcitor_recorder.append(
                            check_if_predicted_the_same_as_gt(actions_scores)
                        )

                    def select_the_best(actions_scores):
                        max_key = max(
                            actions_scores,
                            key=lambda k: actions_scores[k]["predicted_dense_reward"],
                        )

                        if self.save_replan_data is not None:
                            cloned_actions_scores = [
                                {
                                    "actions": x["actions"],
                                    "images_list": x["actions"],
                                    "global_score": x["global_score"],
                                    "steps": x["steps"],
                                    "done": x["done"],
                                    "state": x["state"],
                                    "after_action_states": x["after_action_states"],
                                    "max_one": ac_i == max_key,
                                    "encoded_states": self.states_convetor.get_state_by_current_obs_states(
                                        x["state"]
                                    ),
                                }
                                for ac_i, x in zip(
                                    actions_scores.keys(), actions_scores.values()
                                )
                            ]
                            save_action_selection_data.append(cloned_actions_scores)
                        return actions_scores[max_key]

                    the_best_one = select_the_best(actions_scores)
                    saved_actions_so_far += the_best_one["actions"]

                    self.env = the_best_one["env"]
                    steps = the_best_one["steps"]
                    if self.replan_times == 4:

                        def merge_images(actions_scores):
                            # 提取所有条目的图片列表
                            all_images = [
                                entry["images_list"]
                                for entry in actions_scores.values()
                            ]
                            # 确保每个条目中都有 4 张图片
                            for images_list in all_images:
                                if len(images_list) != self.replan_times:
                                    raise ValueError(
                                        "每个 actions_scores 条目中的 images_list 需要包含 4 张图片。"
                                    )

                            # 确保图片大小一致
                            height, width, channels = all_images[0][0].shape
                            for images_list in all_images:
                                for img in images_list:
                                    if img.shape != (height, width, channels):
                                        raise ValueError("所有图片需要有相同的尺寸以进行拼接。")

                            # 拼接每个图片列表的图片为 2x2
                            merged_images = []
                            for i in range(self.replan_times):  # 针对每张 2x2 的目标图片
                                top_row = np.hstack(
                                    (all_images[0][i], all_images[1][i])
                                )
                                bottom_row = np.hstack(
                                    (all_images[2][i], all_images[3][i])
                                )
                                merged_image = np.vstack((top_row, bottom_row))
                                merged_images.append(merged_image)

                            return merged_images

                        images_list += merge_images(actions_scores)
                    else:
                        images_list += the_best_one["images_list"]

                    def save_trajectories(data_for_save_trajectory):
                        for (
                            steps,
                            states,
                            rew,
                            done,
                            action,
                        ) in data_for_save_trajectory:
                            step_data = {
                                "red_name": "model_1",
                                "blue_name": "model_2",
                                "time": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                                "id": id,
                                "step": steps,
                                "states": states,
                                "rew": rew,
                                "done": done,
                                "action": action,
                                "test_mode": False,
                            }
                            steps_data.append(step_data)

                    save_trajectories(the_best_one["data_for_save_trajectory"])
                    if (
                        the_best_one["global_score"] >= 0.999
                        or the_best_one["done"]
                        or the_best_one["steps"] >= max_steps
                    ):
                        break
                    # img_obs = self.env.render("rgb_array")

                    # pyglet.clock.tick()  # pytype: disable=module-attr
                    # delta = time.time() - last_time
                    # time.sleep(max(0, dt - delta))
                    # last_time = time.time()
                if save_videos:
                    file_name = os.path.join(dir_path, f"{id}_{et}.mp4")
                    with imageio.get_writer(file_name, fps=10) as video:
                        for img in images_list:
                            video.append_data(img)
                    append_step_to_file(json_file_path, steps_data)

                if self.save_replan_data is not None:
                    time_id = formatted_time_with_ms.replace("/", "_")
                    file_name = os.path.join(
                        self.save_replan_data, f"{id}_{et}_{time_id}.npy"
                    )
                    np.save(file_name, save_action_selection_data)
                evaluation_results[str(id)][str(et)] = {
                    "reward": self.env.get_reward(),
                    "global_reward": self.env.global_score(),
                    "global_dense_reward": self.env.global_dense_reward(),
                    "steps": steps,
                    "overlapped_items": self.env.get_overlapped_items_by_category(),
                    "predictor": [
                        len(self.predcitor_recorder),
                        sum(self.predcitor_recorder),
                        sum(self.predcitor_recorder)
                        / (len(self.predcitor_recorder) + 1e-5),
                    ],
                }
                print(evaluation_results)
                end_time = datetime.now()

                print(
                    f"run {et}-th evaluation: start at:"
                    + start_time.strftime("%Y-%m-%d %H:%M:%S")
                    + "end at:"
                    + end_time.strftime("%Y-%m-%d %H:%M:%S")
                )
            average_results = calculate_average_results(evaluation_results[str(id)])
            evaluation_results[str(id)]["average"] = average_results
        if save_videos:
            path = os.path.join(dir_path, "evaluation_results.json")
            with open(path, "w") as f:
                json.dump(evaluation_results, f)

        return evaluation_results
