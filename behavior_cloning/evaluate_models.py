import argparse

import torch

from evaluator import MovingOutEvaluator
from policies import load_policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    ids,
    model_1,
    model_2,
    evaluation_times,
    file_name,
    max_item_number,
    max_steps=500,
    robust_evaluation=False,
    experiment_save_path=None,
    save_videos=True,
):
    evaluator = MovingOutEvaluator(
        map_name=ids[0],
        max_item_number=max_item_number,
        add_noise_to_item=robust_evaluation,
        experiment_save_path=experiment_save_path,
    )
    evaluation_results = evaluator.evaluate_ids(
        ids,
        model_1,
        model_2,
        evaluate_times=evaluation_times,
        max_steps=max_steps,
        save_videos=save_videos,
        file_name=file_name,
    )
    print(evaluation_results)
    return evaluation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate two behavior-cloning agents (any mix of mlp | gru | dp) "
        "in the MovingOut environment."
    )
    parser.add_argument(
        "--map_name", type=str, default="HandOff", nargs="+",
        help="map name(s) to evaluate on (e.g. HandOff)",
    )
    parser.add_argument("--robot_1_model_path", type=str, required=True, help="robot 1 model .pt")
    parser.add_argument("--robot_1_arch", type=str, default="mlp", choices=["mlp", "gru", "dp"])
    parser.add_argument("--robot_2_model_path", type=str, required=True, help="robot 2 model .pt")
    parser.add_argument("--robot_2_arch", type=str, default="mlp", choices=["mlp", "gru", "dp"])
    parser.add_argument("--evaluation_times", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_items_number", type=int, default=7)
    parser.add_argument("--action_type", type=str, default="fb_cos_sin")
    parser.add_argument("--selected_actions", type=int, default=None,
                        help="how many predicted actions to execute per inference (dp)")
    parser.add_argument("--dp_hold_scale", type=float, default=5.0,
                        help="dp only: temperature on the hold logits "
                             "(softmax(logits*scale)); higher = more decisive "
                             "grab/release, <=0 = argmax (deterministic hold)")
    parser.add_argument("--robust_evaluation", action="store_true",
                        help="add noise to item positions at reset")
    parser.add_argument("--experiment_save_path", type=str, default="temp")
    parser.add_argument("--no_videos", action="store_true")

    args = parser.parse_args()

    ids = args.map_name if isinstance(args.map_name, list) else [args.map_name]

    hold_scale = args.dp_hold_scale if args.dp_hold_scale > 0 else None
    model_1 = load_policy(
        args.robot_1_arch, args.robot_1_model_path,
        action_type=args.action_type, selected_actions=args.selected_actions,
        hold_scale=hold_scale,
    )
    model_2 = load_policy(
        args.robot_2_arch, args.robot_2_model_path,
        action_type=args.action_type, selected_actions=args.selected_actions,
        hold_scale=hold_scale,
    )

    file_name = [str(args.robot_1_model_path), str(args.robot_2_model_path)]
    main(
        ids,
        model_1,
        model_2,
        args.evaluation_times,
        file_name,
        max_item_number=args.max_items_number,
        max_steps=args.max_steps,
        robust_evaluation=args.robust_evaluation,
        experiment_save_path=args.experiment_save_path,
        save_videos=not args.no_videos,
    )
