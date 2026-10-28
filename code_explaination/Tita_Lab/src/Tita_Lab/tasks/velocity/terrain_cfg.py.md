# terrain_cfg.py：集中管理可切换的地形方案

**阅读目标：** 理解地形定义、Scene选择与回合升降级的分工；两套配置分别保留官方混合方案和TITA正负台阶。

源码目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity`。
源码文件：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/terrain_cfg.py`。
更新：2026-10-23。当前两套配置均已写入并通过validate，当前Scene已选GYM_TITA_ROUGH_TERRAINS_CFG；课程已接线，见20。地形配置放在velocity任务共用层，config/Tita/env_cfg.py选择方案；这是按职责的项目组织建议，不是Lab强制路径。TITA候选配置validate及原生CPU小规模生成通过；LAB_ROUGH提取沿用官方copy接口，已接入Scene模块引用，未在本轮运行Sim。

```python
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
```

## 1. 为什么放在velocity层

本文件集中保存Lab混合地形和TITA台阶两套生成器配置，几何定义不依赖机器人关节/执行器。机器人专用env_cfg负责选择其中一套，再指定地面路径、材质、初始出生难度与传感器。放在tasks/velocity/terrain_cfg.py可以与同层的mdp和环境类一起供该任务的机器人配置使用，无需新增包或.pyi。

官方本地core/velocity/velocity_env_cfg.py从独立isaaclab.terrains.config.rough导入预设，支持将地形定义与机器人环境配置分开；项目路径是据此作出的职责选择，不声称官方要求外部工程必须使用同一目录。

## 2. Gym原来生成什么

TitaConstraintRoughCfg.terrain继承通用配置：10行20列，每块8×8m，外边界25m，水平网格0.1m、垂直网格0.005m，curriculum=True。utils/terrain.py的其他类型分支均被注释，实际用choice<0.5产生负阶高，其余正阶高；每阶高度为±0.5*(0.05+0.18d)=±(0.025+0.09d)m，阶宽0.61m，平台参数3m。地形类型比例旧字段不再参与这个最终分支。

旧curiculum()中，d=row/10；choice=col/20+0.001。因此前十列向中心降低，后十列向中心升高。

## 3. Lab怎样组成配置

TerrainGeneratorCfg是整张地图的配置对象。size指定一块尺寸；num_rows/num_cols指定拼图数量；border_width指定整张地图的外围边界。sub_terrains是有序字典，键是本项目起的名称，值是Lab已有的子地形配置实例。先写stairs_down使其占前半列，stairs_up占后半列。

HfInvertedPyramidStairsTerrainCfg继承HfPyramidStairsTerrainCfg并令inverted=True，生成函数会将阶高取负。Hf表示先建立二维高度数组，再由包装函数转成三角网格；当前PhysX仍使用生成的网格，无需改变terrain_type="generator"。

阶高范围(0.025,0.115)来自线性插值：0.025+d*(0.115-0.025)。d=0.5时名义阶高0.07m。step_width为一阶水平踏面的宽度；platform_width控制中央平区参数，受离散和整阶布局影响，不保证实际边长精确等于3m。

horizontal_scale使名义阶宽0.61m离散为int(0.61/0.1)=6格，即0.6m；vertical_scale使阶高按5mm取整。slope_threshold用于高度场转三角网格时的陡面修正，不是机器人爬坡角度限制。

## 4. 生成与训练升降级分开接入

curriculum=True使地图难度按行排列，地图在场景创建时生成。Lab原生d=(row+u)/10，u∈[0,1)，与旧固定row/10存在行内扰动差异。当前先采用原生规则，保留该差异，不新增自定义生成器。

Scene保留max_init_terrain_level=5，出生时选行编号0至5；20列按环境编号分配。单环境通常分到第0列，即向下台阶，不能只看这一台便判断图里没有向上台阶。

训练中根据表现升降行号，必须另接CurriculumCfg/term；仅设置生成器curriculum=True不会自动建立该规则。地图定义与初始出生难度接口已写；当前升降级已按[20](../../../../../../docs/migration/20_地形课程迁移.md)接入；尚无完整长运行日志。

## 5. 怎样接到env_cfg

删除ROUGH_TERRAINS_CFG导入，新增：

```python
from ... import terrain_cfg
```

在已有TerrainImporterCfg中改为：

```python
terrain_generator=terrain_cfg.GYM_TITA_ROUGH_TERRAINS_CFG,
```

当前选择GYM_TITA_ROUGH_TERRAINS_CFG；需要官方混合地形对照时，可切换为LAB_TITA_ROUGH_TERRAINS_CFG。原terrain_type、max_init_terrain_level、地形prim路径、材质、机器人和扫描器配置继续使用。这是直接导入配置模块，不经过mdp的lazy_export，无需新增.pyi。

## 6. 验证范围和已知差异

原生配置validate、六个CPU正/负样本、原生2×2生成器通过，见[证据](../../../../../../docs/audits/2026-10-23_tita_terrain/README.md)。没有启动Sim或训练，当前两套配置与课程均已写入；真实长运行尚未提供完整日志。

Lab的行内扰动、边缘填充使网格和原点不保证逐点等价。实际CPU样本d=0.5时正台阶origin_z=0.35m、负台阶origin_z=-0.28m，而负地形最低点约-0.35m；origin是中央2×2m范围的最高高度，不是无条件读取正中心最低点。这项差异保留，当前不为完全等价重写生成器。

本地阅读顺序：hf_terrains_cfg.py的HfPyramidStairsTerrainCfg/HfInvertedPyramidStairsTerrainCfg → hf_terrains.py的pyramid_stairs_terrain → terrain_generator.py的_generate_curriculum_terrains。官方API：[Isaac Lab 3.0 terrains](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.terrains.html)。

## 补充问答

**Q：** 为什么不和tita.py同级？我不知道啊，尽量看看官方有没有选择。

**A：** 本轮进一步检查了本地官方资产与任务源码，并打开官方GitHub的对应文件。官方共享地形位于[isaaclab/terrains/config/rough.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/config/rough.py)，机器人配置位于[isaaclab_assets/robots/unitree.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/unitree.py)；本地[Spot flat_env_cfg.py:65](/home/eaziel/IsaacLab-main/source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/spot/flat_env_cfg.py)则直接定义COBBLESTONE_ROAD_CFG。这些例子体现“共享配置独立放、任务专用配置可以留在任务中”，没有给外部工程指定唯一目录。当前只有velocity任务需要本项目的几套生成方案，所以维持tasks/velocity/terrain_cfg.py的建议；这是项目职责选择，并不表示放assets/terrain_cfg.py与tita.py同级会导致API错误。前面不应把一个可选布局说成官方标准，本次不再调整拟建路径。
