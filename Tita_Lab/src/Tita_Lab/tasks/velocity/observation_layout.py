# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping

import torch


def pack_tita_observations(
    observations: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Assemble the task's 586-feature training input with final clipping."""

    layout = (
        ("policy", 33),
        ("scan", 187),
        ("privileged_state", 6),
        ("dynamics", 30),
        ("history", 330),
    )

    blocks = []
    for name, size in layout:
        value = observations[name]
        if value.ndim != 2 or value.shape[1] != size:
            raise ValueError(f"Expected '{name}' with shape (N, {size}), got {tuple(value.shape)}.")
        blocks.append(value)

    # Internal observation/history tensors retain their unclipped values.
    return torch.cat(blocks, dim=-1).clamp(-100, 100)
