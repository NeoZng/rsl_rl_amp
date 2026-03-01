# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP extension module integrated with the on-policy runner."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
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
        model_cfg: dict[str, Any],
        device: str,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
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
            self.backbone = MLP(
                input_dim=input_dim,
                output_dim=1,
                hidden_dims=hidden_dims,
                activation=activation,
            )
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

    def compute_loss(
        self, expert_sequences: torch.Tensor, policy_sequences: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        expert_scores = self.forward(expert_sequences)
        policy_scores = self.forward(policy_sequences)

        # LSGAN discriminator loss:
        #   E[(D(expert)-1)^2] + E[(D(policy)+1)^2]
        expert_loss = torch.square(expert_scores - 1.0).mean()
        policy_loss = torch.square(policy_scores + 1.0).mean()
        loss = expert_loss + policy_loss

        metrics = {
            "amp/discriminator_loss": loss.detach().item(),
            "amp/expert_loss": expert_loss.detach().item(),
            "amp/policy_loss": policy_loss.detach().item(),
        }
        return loss, metrics

    def gradient_penalty(self, expert_sequences: torch.Tensor) -> torch.Tensor:
        """R1-style penalty on expert inputs for LSGAN regularization."""
        expert_in = expert_sequences.detach().requires_grad_(True)

        expert_scores = self.forward(expert_in)

        expert_grad = torch.autograd.grad(
            outputs=expert_scores.sum(), inputs=expert_in, create_graph=True
        )[0]

        # E[||grad D(phi)||^2] term (the caller applies w_gp / 2).
        return expert_grad.square().sum(dim=-1).mean()

    def predict_reward(self, sequences: torch.Tensor) -> torch.Tensor:
        # r^S = max(0, 1 - 0.25 * (D - 1)^2)
        scores = self.forward(sequences)
        rewards = 1.0 - 0.25 * torch.square(scores - 1.0)
        return torch.clamp(rewards, min=0.0)


class AMPModule:
    """Adversarial Motion Prior module for runner-level integration."""

    _METRIC_KEYS = (
        "amp/discriminator_loss",
        "amp/expert_loss",
        "amp/policy_loss",
        "amp/grad_penalty",
    )

    def __init__(
        self, env: Any, obs: TensorDict, amp_cfg: dict[str, Any], device: str = "cpu"
    ) -> None:
        self.device = torch.device(device)

        self.obs_group = amp_cfg.get("obs_group", "amp")
        if self.obs_group not in obs:
            raise ValueError(
                f"AMP obs group '{self.obs_group}' not found. Available groups: {list(obs.keys())}"
            )

        self.transition_frames = int(amp_cfg.get("transition_frames", 4))
        self.reward_weight = float(amp_cfg.get("reward_weight", 1.0))
        self.reward_warmup_updates = int(amp_cfg.get("reward_warmup_updates", 50))
        env_obj = env.unwrapped if hasattr(env, "unwrapped") else env
        self.step_dt = float(env_obj.step_dt)
        self.discriminator_updates = int(amp_cfg.get("discriminator_updates", 4))
        self.discriminator_batch_size = int(
            amp_cfg.get("discriminator_batch_size", 256)
        )
        self.discriminator_grad_penalty_weight = float(
            amp_cfg.get("discriminator_grad_penalty_weight", 5.0)
        )
        if self.transition_frames <= 0:
            raise ValueError(
                f"transition_frames must be positive, got {self.transition_frames}."
            )
        if self.discriminator_updates <= 0:
            raise ValueError(
                f"discriminator_updates must be positive, got {self.discriminator_updates}."
            )
        if self.discriminator_batch_size <= 0:
            raise ValueError(
                f"discriminator_batch_size must be positive, got {self.discriminator_batch_size}."
            )
        if self.reward_warmup_updates < 0:
            raise ValueError(
                f"reward_warmup_updates must be non-negative, got {self.reward_warmup_updates}."
            )
        self.is_multi_gpu = bool(amp_cfg.get("multi_gpu_cfg", None))
        if self.is_multi_gpu:
            multi_gpu_cfg = amp_cfg["multi_gpu_cfg"]
            self.gpu_world_size = int(multi_gpu_cfg["world_size"])
        else:
            self.gpu_world_size = 1

        amp_obs_dim = int(obs[self.obs_group].shape[-1])
        self.sequence_dim = self.transition_frames * amp_obs_dim

        dataset_builder = resolve_callable(amp_cfg["dataset_builder"])
        dataset_kwargs = dict(amp_cfg.get("dataset_builder_kwargs", {}))
        dataset = dataset_builder(
            env=env, amp_cfg=amp_cfg, device=str(self.device), **dataset_kwargs
        )

        self.expert_sequences = [seq.to(self.device) for seq in dataset["sequences"]]
        self.expert_lengths = torch.as_tensor(
            dataset["lengths"], device=self.device, dtype=torch.long
        )
        if len(self.expert_sequences) == 0:
            raise ValueError("AMP dataset is empty.")

        dataset_obs_dim = int(self.expert_sequences[0].shape[-1])
        if dataset_obs_dim != amp_obs_dim:
            raise ValueError(
                f"AMP obs dim mismatch: env '{self.obs_group}' is {amp_obs_dim}, dataset is {dataset_obs_dim}."
            )

        self._eligible_expert_idx = torch.nonzero(
            self.expert_lengths >= self.transition_frames, as_tuple=False
        ).squeeze(-1)
        if self._eligible_expert_idx.numel() == 0:
            raise ValueError(
                f"No expert sequence has enough frames for transition_frames={self.transition_frames}."
            )

        num_expert_sequences = len(self.expert_sequences)
        expert_max_len = int(self.expert_lengths.max().item())
        self.expert_padded = torch.zeros(
            num_expert_sequences,
            expert_max_len,
            amp_obs_dim,
            device=self.device,
        )
        for i, seq in enumerate(self.expert_sequences):
            self.expert_padded[i, : seq.shape[0]] = seq

        # Sample sequence IDs proportional to the number of valid windows per sequence.
        eligible_windows = (
            self.expert_lengths[self._eligible_expert_idx] - self.transition_frames + 1
        ).float()
        self._eligible_probs = eligible_windows / eligible_windows.sum()

        model_cfg = dict(amp_cfg.get("model_cfg", {}))
        self.discriminator = _AMPDiscriminator(
            input_dim=self.sequence_dim,
            transition_frames=self.transition_frames,
            obs_dim=amp_obs_dim,
            model_cfg=model_cfg,
            device=str(self.device),
        )
        self._warmup_normalization_from_offline_data()

        discriminator_lr = float(amp_cfg.get("discriminator_lr", 1.0e-4))
        self.optimizer = optim.Adam(
            self.discriminator.parameters(), lr=discriminator_lr
        )

        replay_buffer_size = int(amp_cfg.get("replay_buffer_size", 100000))
        self.buffer = AMPRolloutBuffer(
            replay_buffer_size, self.sequence_dim, device=str(self.device)
        )

        self.frame_buffer = torch.zeros(
            env.num_envs, self.transition_frames, amp_obs_dim, device=self.device
        )
        self.frame_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=self.device
        )
        self.write_idx = 0
        order = torch.arange(self.transition_frames, device=self.device)
        self.order_lut = torch.stack(
            [
                torch.roll(order, shifts=-(k + 1), dims=0)
                for k in range(self.transition_frames)
            ],
            dim=0,
        )
        self.window_offsets = torch.arange(
            self.transition_frames, device=self.device
        ).unsqueeze(0)
        self._amp_update_counter = 0

    def broadcast_parameters(self) -> None:
        if not self.is_multi_gpu:
            return
        params = [self.discriminator.state_dict()]
        dist.broadcast_object_list(params, src=0)
        self.discriminator.load_state_dict(params[0])

    @classmethod
    def _zero_metrics(cls) -> dict[str, float]:
        return {key: 0.0 for key in cls._METRIC_KEYS}

    def _reduce_gradients(self) -> None:
        if not self.is_multi_gpu:
            return
        grads = [
            p.grad.view(-1)
            for p in self.discriminator.parameters()
            if p.grad is not None
        ]
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

    def _reduce_discriminator_normalizer(self) -> None:
        if not self.is_multi_gpu or not self.discriminator.state_normalization:
            return

        normalizer = self.discriminator.normalizer
        dist.all_reduce(normalizer._mean, op=dist.ReduceOp.SUM)
        normalizer._mean /= self.gpu_world_size
        dist.all_reduce(normalizer._var, op=dist.ReduceOp.SUM)
        normalizer._var /= self.gpu_world_size
        normalizer._std = torch.sqrt(torch.clamp(normalizer._var, min=0.0))
        dist.all_reduce(normalizer.count, op=dist.ReduceOp.SUM)
        normalizer.count //= self.gpu_world_size

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
        starts = torch.floor(
            torch.rand(batch_size, device=self.device) * (max_starts + 1).float()
        ).long()
        time_idx = starts.unsqueeze(1) + self.window_offsets
        seq_batch = self.expert_padded[seq_ids.unsqueeze(1), time_idx]
        return seq_batch.reshape(batch_size, self.sequence_dim)

    def _warmup_normalization_from_offline_data(self) -> None:
        """Initialize normalization from all offline expert frames.

        For normalization statistics, we only need per-dimension moments. Therefore each
        frame is expanded to transition-length input by repeating itself across all frame
        slots (instead of building temporal windows), so every frame contributes equally.
        """
        if not self.discriminator.state_normalization:
            return
        with torch.no_grad():
            for seq, seq_len in zip(
                self.expert_sequences, self.expert_lengths.tolist(), strict=False
            ):
                if seq_len <= 0:
                    continue
                frame_inputs = seq[:seq_len].repeat(1, self.transition_frames)
                self.discriminator.update_normalizer(frame_inputs)

    def process_transition(
        self, obs_tp1: TensorDict, dones: torch.Tensor
    ) -> torch.Tensor:
        """Process one env transition and return AMP reward tensor."""
        frames = obs_tp1[self.obs_group].to(self.device)
        dones_bool = dones.view(-1).bool().to(self.device)

        # Prevent cross-episode contamination.
        self.frame_buffer[dones_bool] = 0.0
        self.frame_count[dones_bool] = 0

        self.frame_buffer[:, self.write_idx] = frames
        self.frame_count = torch.clamp(self.frame_count + 1, max=self.transition_frames)

        rewards = torch.zeros(frames.shape[0], device=self.device)
        valid_mask = (self.frame_count >= self.transition_frames) & (~dones_bool)
        order = self.order_lut[self.write_idx]
        valid_flat_seq = self.frame_buffer[valid_mask][:, order, :].reshape(
            -1, self.sequence_dim
        )

        if valid_flat_seq.numel() > 0:
            with torch.no_grad():
                rewards[valid_mask] = self.discriminator.predict_reward(valid_flat_seq)
            self.buffer.add(valid_flat_seq)

        self.write_idx = (self.write_idx + 1) % self.transition_frames

        if self.reward_warmup_updates == 0:
            reward_scale = 1.0
        else:
            reward_scale = min(1.0, self._amp_update_counter / self.reward_warmup_updates)
        amp_reward = self.reward_weight * self.step_dt * rewards * reward_scale
        return amp_reward

    def update(self) -> dict[str, float]:
        effective_batch_size = min(len(self.buffer), self.discriminator_batch_size)
        if effective_batch_size == 0:
            print("[WARN] AMP update skipped because effective_batch_size is 0.")
            return self._zero_metrics()

        total = self._zero_metrics()
        with torch.no_grad():
            policy_for_norm = self.buffer.sample(effective_batch_size)
            expert_for_norm = self._sample_expert_sequences(effective_batch_size)
            self.discriminator.update_normalizer(policy_for_norm)
            self.discriminator.update_normalizer(expert_for_norm)
        self._reduce_discriminator_normalizer()

        for _ in range(self.discriminator_updates):
            expert = self._sample_expert_sequences(effective_batch_size)
            policy = self.buffer.sample(effective_batch_size)

            loss, metrics = self.discriminator.compute_loss(
                expert_sequences=expert, policy_sequences=policy
            )
            grad_penalty = torch.tensor(0.0, device=self.device)
            if self.discriminator_grad_penalty_weight > 0.0:
                grad_penalty = self.discriminator.gradient_penalty(
                    expert_sequences=expert
                )
                loss = (
                    loss + 0.5 * self.discriminator_grad_penalty_weight * grad_penalty
                )
            metrics["amp/grad_penalty"] = grad_penalty.detach().item()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._reduce_gradients()
            self.optimizer.step()

            for key in total:
                total[key] += metrics[key]

        for key in total:
            total[key] /= self.discriminator_updates

        for key in total:
            total[key] = self._mean_scalar(total[key])

        self._amp_update_counter += 1
        return total

    def state_dict(self) -> dict[str, Any]:
        return {
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "frame_buffer": self.frame_buffer,
            "frame_count": self.frame_count,
            "write_idx": self.write_idx,
            "buffer_data": self.buffer.data,
            "buffer_position": self.buffer.position,
            "buffer_size": self.buffer.size,
            "amp_update_counter": self._amp_update_counter,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with torch.inference_mode():
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
                self.buffer.position = (
                    int(state.get("buffer_position", 0)) % self.buffer.buffer_size
                )
                self.buffer.size = int(
                    min(state.get("buffer_size", 0), self.buffer.buffer_size)
                )
        if "amp_update_counter" in state:
            self._amp_update_counter = max(int(state["amp_update_counter"]), 0)
