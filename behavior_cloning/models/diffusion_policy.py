import logging
import math
import time
from datetime import datetime
from typing import Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.out_channels = out_channels
        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond=None):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        if cond is not None:
            embed = self.cond_encoder(cond)
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim=None,
        use_obs_encoder=True,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
    ):
        super().__init__()

        dsed = diffusion_step_embed_dim
        all_dims = [dsed] + list(down_dims)
        start_dim = down_dims[0]
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 2),
            nn.Mish(),
            nn.Linear(dsed * 2, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            if use_obs_encoder:
                cond_dim += dsed
                self.global_cond_encoder = nn.Linear(global_cond_dim, dsed)
            else:
                cond_dim += global_cond_dim
                self.global_cond_encoder = nn.Identity()
        self.input_encoder = nn.Linear(input_dim, dsed)
        self.input_decoder = nn.Linear(dsed, input_dim)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, dsed, 1),
        )

        # self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        self.dropout = nn.Dropout(p=0.05)
        print(
            "number of parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: torch.Tensor,
        # return_recon=False,
        **kwargs,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = self.input_encoder(sample)
        sample = einops.rearrange(sample, "b h t -> b t h")
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if len(global_cond.shape) == 3:
            global_cond = global_cond.flatten(start_dim=1)
        global_cond_enc = self.global_cond_encoder(global_cond)
        global_cond_enc = self.dropout(global_cond_enc)
        global_feature = torch.cat([global_feature, global_cond_enc], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        x = self.input_decoder(x)
        return x


class evaluator_dp:
    def __init__(
        self, model_path, previous_steps, selected_actions=2, max_items_number=7
    ) -> None:
        self.loaded_data = torch.load(model_path, weights_only=False)
        self.model = self.loaded_data["model"]
        self.model = self.model.eval()
        self.training_options = self.loaded_data["training_options"]
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        #### please change the code so you get this attributes from torch.load()
        num_inference_steps = 32
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_inference_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        self.noise_scheduler.set_timesteps(num_inference_steps)
        self.pred_horizon = 8
        self.action_dim = 5
        self.max_items_number = max_items_number
        self.selected_actions = selected_actions
        self.previous_steps = previous_steps

    def get_input(self, states, past_states):
        states = torch.tensor(states, dtype=torch.float32).to(states.device).unsqueeze(0)
        past_states = (
            torch.tensor(past_states, dtype=torch.float32).to(states.device).unsqueeze(0)
        )

        return states, past_states

    def get_action(self, states, past_states):
        print("time1", time.time())
        states, past_states = self.get_input(states, past_states)
        states = states.unsqueeze(0)
        if self.previous_steps >= 1:
            states = torch.cat([past_states, states], dim=2)
        else:
            states = states
        # states = torch.cat((past_states, states), dim = 2)
        action = torch.empty(
            (1, self.pred_horizon, self.action_dim), device=states.device
        ).normal_(-1, 1)
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.model(action, k, states)
            action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=action
            )[0]

        move_actions = []
        hold_actions = []
        action_type = "fb_cos_sin"
        selected_action = self.selected_actions
        for i in range(selected_action):
            hold_logit = action[0, i : i + 1, 3:]

            def de_normalize_action(normalized_action):
                x_component = normalized_action[:, :, 0]
                y_component = normalized_action[:, :, 1]
                speed = torch.sqrt(x_component**2 + y_component**2)
                angle = torch.atan2(y_component, x_component)
                de_normalized_action = torch.cat(
                    [speed.unsqueeze(2), angle.unsqueeze(2)], dim=2
                )
                return de_normalized_action

            ac_dis = action[:, i : i + 1, 1:3]

            ac_dis = de_normalize_action(ac_dis).squeeze()
            ac_dis[0] = action[:, i : i + 1, 0]
            probabilities = F.softmax(hold_logit * 5, dim=-1)
            holding_action = torch.multinomial(probabilities, num_samples=1)
            holding_action = holding_action.squeeze(1)

            move_actions.append(ac_dis)
            hold_actions.append(holding_action)
        print("time2", time.time())
        return move_actions, hold_actions

