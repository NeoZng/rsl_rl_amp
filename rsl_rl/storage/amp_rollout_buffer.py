# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP rollout buffer for policy-generated N-frame sequences."""

from __future__ import annotations

import torch


class AMPRolloutBuffer:
    """Circular buffer for flattened AMP sequences."""

    def __init__(self, buffer_size: int, sequence_dim: int, device: str = "cpu") -> None:
        self.buffer_size = int(buffer_size)
        self.sequence_dim = int(sequence_dim)
        self.device = torch.device(device)

        self.data = torch.zeros(self.buffer_size, self.sequence_dim, device=self.device)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, sequences: torch.Tensor) -> None:
        if sequences.numel() == 0:
            return
        sequences = sequences.to(self.device)
        if sequences.ndim != 2 or sequences.shape[1] != self.sequence_dim:
            raise ValueError(
                f"Expected sequences with shape (batch, {self.sequence_dim}), got {tuple(sequences.shape)}"
            )

        batch_size = sequences.shape[0]
        if batch_size >= self.buffer_size:
            self.data.copy_(sequences[-self.buffer_size :])
            self.position = 0
            self.size = self.buffer_size
            return

        end = self.position + batch_size
        if end <= self.buffer_size:
            self.data[self.position : end] = sequences
        else:
            first_chunk = self.buffer_size - self.position
            self.data[self.position :] = sequences[:first_chunk]
            self.data[: end % self.buffer_size] = sequences[first_chunk:]

        self.position = end % self.buffer_size
        self.size = min(self.size + batch_size, self.buffer_size)

    def sample(self, batch_size: int) -> torch.Tensor:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty AMP rollout buffer.")
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.data[idx]

    def clear(self) -> None:
        self.position = 0
        self.size = 0
        self.data.zero_()
