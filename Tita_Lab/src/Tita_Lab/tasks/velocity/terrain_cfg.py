# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain configurations for the velocity task."""

from isaaclab.terrains import (
    HfInvertedPyramidStairsTerrainCfg,
    HfPyramidStairsTerrainCfg,
    TerrainGeneratorCfg,
)
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

# 创建独立配置，保留当前Lab混合地形的全部参数。
LAB_TITA_ROUGH_TERRAINS_CFG = ROUGH_TERRAINS_CFG.copy()

GYM_TITA_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),  # 每块地形的x/y尺寸，单位m。
    num_rows=10,  # 行编号0~9，对应由低到高的难度。
    num_cols=20,  # 前10列向下台阶，后10列向上台阶。
    border_width=25.0,  # 整张地形外围的边界宽度，单位m。
    horizontal_scale=0.1,  # 高度场xy网格间距，单位m。
    vertical_scale=0.005,  # 每个整数高度单位对应5 mm。
    slope_threshold=0.75,
    curriculum=True,  # 按行排列难度
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "stairs_down": HfInvertedPyramidStairsTerrainCfg(
            proportion=0.5,
            # |h| = 0.025 + 0.09*difficulty，单位m。
            step_height_range=(0.025, 0.115),
            step_width=0.61,
            platform_width=3.0,
        ),
        "stairs_up": HfPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.025, 0.115),
            step_width=0.61,
            platform_width=3.0,
        ),
    },
)
