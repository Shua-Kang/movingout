import argparse

import torch
from evaluate_two_agents import mlp_warpper
from evaluate_two_gru import gru_wrapper
from models.gru import TrajectoryGRU
from models.mlp import TrajectoryMLP
from moving_out.evaluation import MovingOutEvaluator
from models.diffusion_policy import ConditionalResidualBlock1D, ConditionalUnet1D, Conv1dBlock, Downsample1d, SinusoidalPosEmb, Upsample1d
from train_dp import evaluator_dp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(
    ids,
    model_1,
    model_2,
    evaluation_times,
    precition_horizon,
    file_name,
    max_item_number,
    replan_times,
    model_1_replan_times,
    model_2_replan_times,
    robust_evaluation=False,
    experiment_save_path=None,
    save_replan_data=False,
    ranking_model=None,
    replan_strategy=None,
    another_action_predictor=None,
):
    evaluator = MovingOutEvaluator(
        map_name=ids[0],
        max_item_number=max_item_number,
        add_noise_to_item=robust_evaluation,
        replan_times=replan_times,
        model_1_replan_times=model_1_replan_times,
        model_2_replan_times=model_2_replan_times,
        experiment_save_path=experiment_save_path,
        save_replan_data=save_replan_data,
        ranking_model=ranking_model,
        replan_strategy=replan_strategy,
        another_action_predictor=another_action_predictor,
    )
    evaluation_results = evaluator.evaluate_ids(
        ids,
        model_1,
        model_2,
        evaluate_times=evaluation_times,
        max_steps=500,
        model_horizon=precition_horizon,
        file_name=file_name,
    )
    print(evaluation_results)


if __name__ == "__main__":
    procedure = ""
    parser = argparse.ArgumentParser(description="Process JSON file and ID.")
    parser.add_argument(
        "--id_number", type=int, default=0, nargs="+", help="The ID number to extract"
    )

    parser.add_argument("--model_1_path_robot", type=str, help="model path")
    parser.add_argument("--model_1_arch", type=str, help="model path")
    parser.add_argument("--model_1_replan_strategy", type=str, default=None)
    parser.add_argument("--model_1_replan_times", type=int, default=1)
    parser.add_argument("--model_1_vae_model_path", type=str, default=None)
    parser.add_argument("--model_1_precition_horizon", type=int, default=None)
    parser.add_argument("--model_1_selected_actions", type=int, default=None)

    parser.add_argument("--model_2_path_robot", type=str, help="model path")
    parser.add_argument("--model_2_arch", type=str, help="model path")
    parser.add_argument("--model_2_replan_strategy", type=str, default=None)
    parser.add_argument("--model_2_replan_times", type=int, default=1)
    parser.add_argument("--model_2_vae_model_path", type=str, default=None)
    parser.add_argument("--model_2_precition_horizon", type=int, default=None)
    parser.add_argument("--model_2_selected_actions", type=int, default=None)

    parser.add_argument(
        "--evaluation_times", type=int, default=1, help="evaluation_times"
    )
    parser.add_argument(
        "--max_items_number", type=int, default=7, help="evaluation_times"
    )
    parser.add_argument(
        "--action_type", type=str, default="fb_cos_sin", help="evaluation_times"
    )
    # parser.add_argument("--replan_times", type=int, default=1 )
    parser.add_argument("--robust_evaluation", action="store_true")
    parser.add_argument("--experiment_save_path", type=str, default="temp")
    parser.add_argument("--ranking_model_path", type=str, default=None)
    parser.add_argument("--action_predictor_path", type=str, default=None)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    if args.model_1_arch == "mlp":
        model_1 = mlp_warpper(args.model_1_path_robot, args.action_type)
    elif args.model_1_arch == "gru":
        model_1 = gru_wrapper(args.model_1_path_robot, args.action_type)
    elif args.model_1_arch == "dp":
        model_1 = evaluator_dp(
            args.model_1_path_robot,
            previous_steps=args.model_1_precition_horizon,
            selected_actions=args.model_1_selected_actions,
            max_items_number=args.max_items_number,
        )
    else:
        print("ERROR: no model arch : ", args.model_1_arch)
        exit(0)

    if args.model_2_arch == "mlp":
        model_2 = mlp_warpper(args.model_2_path_robot, args.action_type)
    elif args.model_2_arch == "gru":
        model_2 = gru_wrapper(args.model_2_path_robot, args.action_type)
    elif args.model_2_arch == "dp":
        model_2 = evaluator_dp(
            args.model_2_path_robot,
            previous_steps=args.model_2_precition_horizon,
            selected_actions=args.model_2_selected_actions,
            max_items_number=args.max_items_number,
        )
    else:
        print("ERROR: no model arch : ", args.model_2_arch)
        exit(0)

    ranking_model = None
    another_action_predictor = None
    save_replan_data = None
    replan_strategy = None

    if args.action_predictor_path is not None and args.replan_times > 1:
        another_action_predictor = another_agent_action_predictor(
            args.action_predictor_path,
            previous_steps=1,
            selected_actions=4,
        )

    file_name = [str(args.model_1_path_robot), str(args.model_2_path_robot)]
    main(
        args.id_number,
        model_1,
        model_2,
        args.evaluation_times,
        args.model_1_precition_horizon,
        file_name,
        max_item_number=args.max_items_number,
        replan_times=args.model_1_replan_times,
        model_1_replan_times=args.model_1_replan_times,
        model_2_replan_times=args.model_2_replan_times,
        robust_evaluation=args.robust_evaluation,
        experiment_save_path=args.experiment_save_path,
        save_replan_data=None,
        ranking_model=ranking_model,
        replan_strategy=replan_strategy,
        another_action_predictor=another_action_predictor,
    )
