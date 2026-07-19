"""Inference-time wrappers for the three behavior-cloning agents.

Every policy exposes the same interface used by the evaluator:

    get_action(states, past_states) -> (move_actions, hold_actions)

where move_actions is a list of [forward_backward, angle] pairs and
hold_actions a matching list of 0/1 hold decisions, one per executed step.

- MLPPolicy: deterministic (argmax over the hold logits, regression output
  used as-is).
- GRUPolicy: stochastic - the GRU's initial hidden state is random noise
  (see models/gru.py) and the hold decision is sampled.
- DiffusionPolicy: stochastic - actions are denoised from random noise with
  a DDIM scheduler and the hold decision is sampled.
"""

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensors(states, past_states):
    states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
    past_states = torch.tensor(
        past_states, dtype=torch.float32, device=device
    ).unsqueeze(0)
    return states, past_states


def _decode_fb_cos_sin(mse_output):
    """[1, k, 3] regression output -> list of [fb, angle] rows."""
    fb = torch.clip(mse_output[0, :, 0], -1, 1)
    angle = torch.atan2(mse_output[0, :, 2], mse_output[0, :, 1])
    return torch.stack([fb, angle], dim=1)


class _RegressionPolicy:
    """Shared logic for the MLP and GRU wrappers (same model interface)."""

    deterministic_hold = True

    def __init__(self, model_path, action_type="fb_cos_sin"):
        self.action_type = action_type
        self.loaded_data = torch.load(model_path, weights_only=False, map_location=device)
        self.model = self.loaded_data["model"].to(device)
        self.model = self.model.eval()
        self.training_options = self.loaded_data["training_options"] or {}
        self.selected_actions = self.training_options.get("selected_actions", None)

    @torch.no_grad()
    def get_action(self, states, past_states):
        states, past_states = _to_tensors(states, past_states)
        mse_output, ce_logits, _ = self.model(past_states, states)

        move = _decode_fb_cos_sin(mse_output)  # [k, 2]

        if self.deterministic_hold:
            holds = torch.argmax(ce_logits[0], dim=-1, keepdim=True)  # [k, 1]
        else:
            probabilities = torch.softmax(ce_logits[0], dim=-1)
            holds = torch.multinomial(probabilities, num_samples=1)  # [k, 1]

        k = move.shape[0] if self.selected_actions is None else min(
            self.selected_actions, move.shape[0]
        )
        move_actions = [move[i] for i in range(k)]
        hold_actions = [holds[i] for i in range(k)]
        return move_actions, hold_actions


class MLPPolicy(_RegressionPolicy):
    """Deterministic MLP agent."""

    deterministic_hold = True


class GRUPolicy(_RegressionPolicy):
    """Stochastic GRU agent (random-noise initial hidden state + sampled hold)."""

    deterministic_hold = False


class DiffusionPolicy:
    """Stochastic diffusion-policy agent (DDIM denoising from random noise)."""

    def __init__(self, model_path, selected_actions=None, num_inference_steps=None,
                 hold_scale=5.0):
        # hold_scale sharpens the hold logits before sampling:
        # softmax(logits * hold_scale). Small values -> noisy grab/release,
        # large values -> near-deterministic; None -> argmax (fully
        # deterministic hold). This knob changes grab behavior a lot.
        self.hold_scale = hold_scale
        self.loaded_data = torch.load(model_path, weights_only=False, map_location=device)
        self.model = self.loaded_data["model"].to(device)
        self.model = self.model.eval()
        self.training_options = self.loaded_data["training_options"] or {}

        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        if num_inference_steps is None:
            num_inference_steps = self.training_options.get("num_train_timesteps", 32)
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_inference_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        self.noise_scheduler.set_timesteps(num_inference_steps)
        self.pred_horizon = self.training_options.get("pred_horizon", 8)
        self.action_dim = self.training_options.get("action_dim", 5)
        self.previous_steps = self.training_options.get("previous_steps", 1)
        if selected_actions is None:
            selected_actions = self.training_options.get("selected_actions", 2)
        self.selected_actions = selected_actions

    @torch.no_grad()
    def get_action(self, states, past_states):
        states, past_states = _to_tensors(states, past_states)
        if self.previous_steps >= 1:
            obs = torch.cat(
                [past_states.reshape(past_states.size(0), -1), states], dim=1
            )
        else:
            obs = states

        action = torch.empty(
            (1, self.pred_horizon, self.action_dim), device=device
        ).normal_(-1, 1)
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.model(action, k, obs)
            action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=action
            )[0]

        # action layout per step: [fb, cos, sin, hold_onehot x2]
        fb = torch.clip(action[0, :, 0], -1, 1)
        angle = torch.atan2(action[0, :, 2], action[0, :, 1])
        move = torch.stack([fb, angle], dim=1)  # [pred_horizon, 2]

        if self.hold_scale is None:
            holds = torch.argmax(action[0, :, 3:], dim=-1, keepdim=True)
        else:
            probabilities = F.softmax(action[0, :, 3:] * self.hold_scale, dim=-1)
            holds = torch.multinomial(probabilities, num_samples=1)  # [pred_horizon, 1]

        k = min(self.selected_actions, move.shape[0])
        move_actions = [move[i] for i in range(k)]
        hold_actions = [holds[i] for i in range(k)]
        return move_actions, hold_actions


def load_policy(arch, model_path, action_type="fb_cos_sin", selected_actions=None,
                hold_scale=5.0):
    """Build the right policy wrapper for a saved model.

    hold_scale (dp only): temperature on the hold logits; higher = more
    decisive grab/release, None = argmax."""
    if arch == "mlp":
        return MLPPolicy(model_path, action_type)
    elif arch == "gru":
        return GRUPolicy(model_path, action_type)
    elif arch == "dp":
        return DiffusionPolicy(model_path, selected_actions=selected_actions,
                               hold_scale=hold_scale)
    raise ValueError(f"unknown model arch {arch!r} (expected mlp | gru | dp)")
