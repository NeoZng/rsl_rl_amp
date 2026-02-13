# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP extension module integrated with the on-policy runner."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from tensordict import TensorDict

from rsl_rl.modules import CNN, MLP, EmpiricalNormalization
from rsl_rl.storage import AMPRolloutBuffer
from rsl_rl.utils import resolve_callable


class _AMPDiscriminator(nn.Module):
    """Configurable discriminator with optional input normalization."""

    def __init__(
        self,
        input_dim: int,
        transition_frames: int,
        obs_dim: int,
        model_cfg: dict,
        device: str,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.transition_frames = transition_frames
        self.obs_dim = obs_dim

        self.state_normalization = bool(model_cfg.get("state_normalization", False))
        self.normalizer = (
            EmpiricalNormalization(shape=[input_dim], until=int(1.0e8)).to(self.device)
            if self.state_normalization
            else nn.Identity()
        )

        class_name = model_cfg.get("class_name", "rsl_rl.modules:MLP")
        resolved = resolve_callable(class_name)

        if resolved is MLP:
            hidden_dims = model_cfg.get("hidden_dims", [512, 256])
            activation = model_cfg.get("activation", "elu")
            self.backbone = MLP(input_dim=input_dim, output_dim=1, hidden_dims=hidden_dims, activation=activation)
            self._forward_impl = self._forward_mlp
        elif resolved is CNN:
            # Treat sequence as a single-channel image: (B, 1, T, obs_dim)
            cnn_cfg = dict(model_cfg.get("cnn_cfg", {}))
            output_channels = cnn_cfg.pop("output_channels", [32, 64])
            kernel_size = cnn_cfg.pop("kernel_size", 3)
            stride = cnn_cfg.pop("stride", 1)
            dilation = cnn_cfg.pop("dilation", 1)
            padding = cnn_cfg.pop("padding", "none")
            norm = cnn_cfg.pop("norm", "none")
            activation = cnn_cfg.pop("activation", "elu")
            max_pool = cnn_cfg.pop("max_pool", False)
            global_pool = cnn_cfg.pop("global_pool", "avg")

            self.backbone = CNN(
                input_dim=(transition_frames, obs_dim),
                input_channels=1,
                output_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                norm=norm,
                activation=activation,
                max_pool=max_pool,
                global_pool=global_pool,
                flatten=True,
            )
            latent_dim = int(self.backbone.output_dim)
            self.head = nn.Linear(latent_dim, 1)
            self._forward_impl = self._forward_cnn
        else:
            raise ValueError(
                f"Unsupported AMP discriminator model class '{class_name}'. "
                "Use 'rsl_rl.modules:MLP' or 'rsl_rl.modules:CNN'."
            )

        self.to(self.device)

    def _forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)

    def _forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        x_img = x.view(x.shape[0], 1, self.transition_frames, self.obs_dim)
        latent = self.backbone(x_img)
        return self.head(latent).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.normalizer(x)
        return self._forward_impl(x)

    def update_normalizer(self, x: torch.Tensor) -> None:
        if self.state_normalization:
            self.normalizer.update(x)  # type: ignore

    def compute_loss(self, expert_sequences: torch.Tensor, policy_sequences: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        stacked = torch.cat([expert_sequences, policy_sequences], dim=0)
        self.update_normalizer(stacked)

        expert_logits = self.forward(expert_sequences)
        policy_logits = self.forward(policy_sequences)

        expert_loss = F.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits))
        policy_loss = F.binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits))
        loss = expert_loss + policy_loss

        metrics = {
            "amp/discriminator_loss": loss.detach().item(),
            "amp/expert_loss": expert_loss.detach().item(),
            "amp/policy_loss": policy_loss.detach().item(),
        }
        return loss, metrics

    def gradient_penalty(self, expert_sequences: torch.Tensor, policy_sequences: torch.Tensor) -> torch.Tensor:
        """R1-style penalty on expert and policy inputs to keep logits smooth."""
        expert_in = expert_sequences.detach().requires_grad_(True)
        policy_in = policy_sequences.detach().requires_grad_(True)

        expert_logits = self.forward(expert_in)
        policy_logits = self.forward(policy_in)

        expert_grad = torch.autograd.grad(
            outputs=expert_logits.sum(),
            inputs=expert_in,
            create_graph=True,
            retain_graph=True,
        )[0]
        policy_grad = torch.autograd.grad(
            outputs=policy_logits.sum(),
            inputs=policy_in,
            create_graph=True,
            retain_graph=True,
        )[0]

        expert_gp = expert_grad.square().sum(dim=-1).mean()
        policy_gp = policy_grad.square().sum(dim=-1).mean()
        return 0.5 * (expert_gp + policy_gp)

    def predict_reward(self, sequences: torch.Tensor) -> torch.Tensor:
        logits = self.forward(sequences)
        return torch.sigmoid(logits)


class AMPModule:
    """Adversarial Motion Prior module for runner-level integration."""

    def __init__(self, env, obs: TensorDict, amp_cfg: dict, device: str = "cpu") -> None:
        self.env = env
        self.device = torch.device(device)
        self.cfg = amp_cfg

        self.obs_group = amp_cfg.get("obs_group", "amp")
        if self.obs_group not in obs.keys():
            raise ValueError(
                f"AMP obs group '{self.obs_group}' not found. Available groups: {list(obs.keys())}"
            )

        self.transition_frames = int(amp_cfg.get("transition_frames", 4))
        self.reward_weight = float(amp_cfg.get("reward_weight", 1.0))
        if hasattr(env, "cfg") and hasattr(env.cfg, "sim") and hasattr(env.cfg, "decimation"):
            self.step_dt = float(env.cfg.sim.dt) * float(env.cfg.decimation)
        else:
            self.step_dt = float(env.unwrapped.step_dt if hasattr(env, "unwrapped") else env.cfg.sim.dt)
        self.discriminator_updates = int(amp_cfg.get("discriminator_updates", 1))
        self.discriminator_batch_size = int(amp_cfg.get("discriminator_batch_size", 1024))
        self.discriminator_grad_penalty_weight = float(amp_cfg.get("discriminator_grad_penalty_weight", 0.0))
        self.is_multi_gpu = bool(amp_cfg.get("multi_gpu_cfg", None))
        if self.is_multi_gpu:
            self.gpu_world_size = int(amp_cfg["multi_gpu_cfg"]["world_size"])
            self.gpu_global_rank = int(amp_cfg["multi_gpu_cfg"]["global_rank"])
        else:
            self.gpu_world_size = 1
            self.gpu_global_rank = 0

        amp_obs_dim = int(obs[self.obs_group].shape[-1])
        self.sequence_dim = self.transition_frames * amp_obs_dim

        dataset_builder = resolve_callable(amp_cfg["dataset_builder"])
        dataset_kwargs = dict(amp_cfg.get("dataset_builder_kwargs", {}))
        dataset = dataset_builder(env=env, amp_cfg=amp_cfg, device=str(self.device), **dataset_kwargs)

        self.expert_sequences = [seq.to(self.device) for seq in dataset["sequences"]]
        self.expert_lengths = torch.as_tensor(dataset["lengths"], device=self.device, dtype=torch.long)
        if len(self.expert_sequences) == 0:
            raise ValueError("AMP dataset is empty.")

        dataset_obs_dim = int(self.expert_sequences[0].shape[-1])
        if dataset_obs_dim != amp_obs_dim:
            raise ValueError(
                f"AMP obs dim mismatch: env '{self.obs_group}' is {amp_obs_dim}, dataset is {dataset_obs_dim}."
            )

        self._eligible_expert_idx = torch.nonzero(self.expert_lengths >= self.transition_frames, as_tuple=False).squeeze(-1)
        if self._eligible_expert_idx.numel() == 0:
            raise ValueError(f"No expert sequence has enough frames for transition_frames={self.transition_frames}.")

        self.num_expert_sequences = len(self.expert_sequences)
        self.expert_max_len = int(self.expert_lengths.max().item())
        self.expert_padded = torch.zeros(
            self.num_expert_sequences, self.expert_max_len, amp_obs_dim, device=self.device
        )
        for i, seq in enumerate(self.expert_sequences):
            self.expert_padded[i, : seq.shape[0]] = seq

        eligible_lengths = self.expert_lengths[self._eligible_expert_idx].float()
        self._eligible_probs = eligible_lengths / eligible_lengths.sum()

        model_cfg = dict(amp_cfg.get("model_cfg", {}))
        self.discriminator = _AMPDiscriminator(
            input_dim=self.sequence_dim,
            transition_frames=self.transition_frames,
            obs_dim=amp_obs_dim,
            model_cfg=model_cfg,
            device=str(self.device),
        )

        discriminator_lr = float(amp_cfg.get("discriminator_lr", 1.0e-3))
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)

        replay_buffer_size = int(amp_cfg.get("replay_buffer_size", 200000))
        self.buffer = AMPRolloutBuffer(replay_buffer_size, self.sequence_dim, device=str(self.device))

        self.frame_buffer = torch.zeros(env.num_envs, self.transition_frames, amp_obs_dim, device=self.device)
        self.frame_count = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.write_idx = 0
        order = torch.arange(self.transition_frames, device=self.device)
        self.order_lut = torch.stack(
            [torch.roll(order, shifts=-(k + 1), dims=0) for k in range(self.transition_frames)], dim=0
        )

        self._valid_frames = 0
        self._total_frames = 0
        self._amp_reward_sum = 0.0
        self._amp_reward_count = 0

    def broadcast_parameters(self) -> None:
        if not self.is_multi_gpu:
            return
        params = [self.discriminator.state_dict()]
        dist.broadcast_object_list(params, src=0)
        self.discriminator.load_state_dict(params[0])

    def _reduce_gradients(self) -> None:
        if not self.is_multi_gpu:
            return
        grads = [p.grad.view(-1) for p in self.discriminator.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for p in self.discriminator.parameters():
            if p.grad is not None:
                n = p.numel()
                p.grad.data.copy_(flat[offset : offset + n].view_as(p.grad.data))
                offset += n

    def _mean_scalar(self, value: float) -> float:
        if not self.is_multi_gpu:
            return value
        tensor = torch.tensor(value, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.gpu_world_size
        return float(tensor.item())

    def _sample_expert_sequences(self, batch_size: int) -> torch.Tensor:
        pick = torch.multinomial(self._eligible_probs, batch_size, replacement=True)
        seq_ids = self._eligible_expert_idx[pick]
        seq_lens = self.expert_lengths[seq_ids]
        max_starts = seq_lens - self.transition_frames
        starts = torch.floor(torch.rand(batch_size, device=self.device) * (max_starts + 1).float()).long()
        offsets = torch.arange(self.transition_frames, device=self.device).unsqueeze(0)
        time_idx = starts.unsqueeze(1) + offsets
        seq_batch = self.expert_padded[seq_ids.unsqueeze(1), time_idx]
        return seq_batch.reshape(batch_size, self.sequence_dim)

    def process_transition(
        self,
        obs_tp1: TensorDict,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Process one env transition and return AMP reward."""
        frames = obs_tp1[self.obs_group].to(self.device)
        dones_bool = dones.view(-1).bool().to(self.device)

        self.frame_buffer[:, self.write_idx] = frames
        self.frame_count = torch.clamp(self.frame_count + 1, max=self.transition_frames)

        rewards = torch.zeros(frames.shape[0], device=self.device)
        valid_mask = self.frame_count >= self.transition_frames
        order = self.order_lut[self.write_idx]
        valid_flat_seq = self.frame_buffer[valid_mask][:, order, :].reshape(-1, self.sequence_dim)

        if valid_flat_seq.numel() > 0:
            with torch.no_grad():
                rewards[valid_mask] = self.discriminator.predict_reward(valid_flat_seq)
            self.buffer.add(valid_flat_seq)

        self.frame_count[dones_bool] = 0
        self.write_idx = (self.write_idx + 1) % self.transition_frames

        self._valid_frames += int(valid_mask.sum().item())
        self._total_frames += int(valid_mask.numel())

        amp_reward = self.reward_weight * self.step_dt * rewards
        self._amp_reward_sum += float(amp_reward.sum().item())
        self._amp_reward_count += int(amp_reward.numel())
        metrics = {
            "valid_ratio": float(valid_mask.float().mean().item()),
            "amp_reward_mean": float(amp_reward.mean().item()),
        }
        return amp_reward, metrics

    def update(self) -> dict[str, float]:
        if len(self.buffer) < self.discriminator_batch_size:
            valid_ratio = (self._valid_frames / self._total_frames) if self._total_frames > 0 else 0.0
            valid_ratio = self._mean_scalar(valid_ratio)
            self._valid_frames = 0
            self._total_frames = 0
            return {
                "amp/discriminator_loss": 0.0,
                "amp/expert_loss": 0.0,
                "amp/policy_loss": 0.0,
                "amp/grad_penalty": 0.0,
                "Episode_Reward/amp_valid_ratio": valid_ratio,
                "Episode_Reward/amp_step_reward_mean": self._mean_scalar(
                    self._amp_reward_sum / max(self._amp_reward_count, 1)
                ),
            }

        total = {
            "amp/discriminator_loss": 0.0,
            "amp/expert_loss": 0.0,
            "amp/policy_loss": 0.0,
            "amp/grad_penalty": 0.0,
        }

        for _ in range(self.discriminator_updates):
            expert = self._sample_expert_sequences(self.discriminator_batch_size).reshape(self.discriminator_batch_size, -1)
            policy = self.buffer.sample(self.discriminator_batch_size)

            loss, metrics = self.discriminator.compute_loss(expert_sequences=expert, policy_sequences=policy)
            grad_penalty = torch.tensor(0.0, device=self.device)
            if self.discriminator_grad_penalty_weight > 0.0:
                grad_penalty = self.discriminator.gradient_penalty(expert_sequences=expert, policy_sequences=policy)
                loss = loss + self.discriminator_grad_penalty_weight * grad_penalty
            metrics["amp/grad_penalty"] = grad_penalty.detach().item()

            self.optimizer.zero_grad()
            loss.backward()
            self._reduce_gradients()
            self.optimizer.step()

            for key in total:
                total[key] += metrics[key]

        for key in total:
            total[key] /= self.discriminator_updates

        total["Episode_Reward/amp_valid_ratio"] = (
            self._valid_frames / self._total_frames
        ) if self._total_frames > 0 else 0.0
        for key in total:
            total[key] = self._mean_scalar(total[key])
        total["Episode_Reward/amp_step_reward_mean"] = self._mean_scalar(
            self._amp_reward_sum / max(self._amp_reward_count, 1)
        )
        self._valid_frames = 0
        self._total_frames = 0
        self._amp_reward_sum = 0.0
        self._amp_reward_count = 0
        return total

    def state_dict(self) -> dict:
        return {
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "frame_buffer": self.frame_buffer,
            "frame_count": self.frame_count,
            "write_idx": self.write_idx,
            "buffer_data": self.buffer.data,
            "buffer_position": self.buffer.position,
            "buffer_size": self.buffer.size,
        }

    def load_state_dict(self, state: dict) -> None:
        self.discriminator.load_state_dict(state["discriminator"])
        if "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:
                # Optimizer state is optional for inference or shape-changed resumes.
                pass

        # Runtime buffers depend on num_envs/replay size and may differ between train/play.
        if "frame_buffer" in state:
            src = state["frame_buffer"].to(self.device)
            if tuple(src.shape) == tuple(self.frame_buffer.shape):
                self.frame_buffer.copy_(src)
        if "frame_count" in state:
            src = state["frame_count"].to(self.device)
            if tuple(src.shape) == tuple(self.frame_count.shape):
                self.frame_count.copy_(src)
        if "write_idx" in state:
            self.write_idx = int(state["write_idx"]) % self.transition_frames

        if "buffer_data" in state:
            src = state["buffer_data"].to(self.device)
            if tuple(src.shape) == tuple(self.buffer.data.shape):
                self.buffer.data.copy_(src)
                self.buffer.position = int(state.get("buffer_position", 0)) % self.buffer.buffer_size
                self.buffer.size = int(min(state.get("buffer_size", 0), self.buffer.buffer_size))
