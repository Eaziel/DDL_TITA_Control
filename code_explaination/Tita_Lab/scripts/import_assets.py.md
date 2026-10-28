# import_assets.py：独立资产与物理循环

内容纯AI生成，由人类要求生成，未经人类修改。当前核对：2026-10-28。

源码：[import_assets.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/scripts/import_assets.py)。本页解释当前已保存代码，脚本未由助手修改或重跑。

## 1. 要验证什么

独立脚本在平地创建TITA，写入起始关节状态、接触材质和执行器目标，再推进物理。它不建立RL的命令、奖励或cost管理器；RL任务另由TitaVelocityEnv创建。

## 2. 启动与资产导入

```python
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from Tita_Lab.assets.tita import TITA_CFG
```

当前文件仍使用AppLauncher兼容入口，它在main中存在但已弃用。这里启动Kit后再使用依赖仿真的接口。新写的actor脚本采用add_launcher_args与launch_simulation；无需为了联调actor而同时重写本文件。

```python
tita_robot = TITA_CFG.replace(prim_path="{ENV_REGEX_NS}/TITA_Robot")
robot = scene["tita_robot"]
```

字段`tita_robot`决定Scene字典的键，prim_path决定USD中的位置。TITA_CFG从当前包的assets/tita.py读取。

## 3. 起始状态与目标

```python
joint_pos = robot.data.default_joint_pos.torch.clone()
joint_vel = robot.data.default_joint_vel.torch.clone()
joint_pos *= torch.empty_like(joint_pos).uniform_(0.5, 1.5)
joint_vel.zero_()
```

默认状态是采样参考，clone得到可以修改的张量。代码随后把角度夹到关节限位，并分别写入实际状态与执行器目标：实际状态决定机器人此刻在哪里，目标决定执行器随后追踪什么。

材质部分生成两个独立的64候选表，为每台机器人选择摩擦和恢复系数，写入该机器人的全部碰撞形状。张量索引、广播和Torch/Warp共享数据解释见[非RL脚本说明](../../TITA_non_RL/scripts/run_sim.py.md)。

## 4. 当前确实推进物理

```python
sim.reset()
run_simulator(sim, scene)
```

`run_simulator`的主循环是：

```python
while simulation_app.is_running():
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)
```

每次先向后端提交目标，再推进一个物理步，最后让Scene按新状态更新数据。`sim.render()`只刷新画面；当前主入口没有停留在过去的静态render诊断。

## 5. 当前运行方式与范围

由用户运行：

```bash
source "$HOME/my_venvs/lab3_sim62/.venv/bin/activate"
cd /mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab
python scripts/import_assets.py --num_envs 1 --device cuda:0 --viz kit
```

脚本使用argparse，不接收RL任务的`physics=...`这种Hydra覆盖语法。运行成功也只能覆盖这个独立场景，不能据此证明RL环境和完整训练已经验证。

官方接口：[启动API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.app.html)。建议维护项见[代码修改清单](../../../docs/当前代码检查与修改清单.md)。

## 补充问答

**Q：** 啥反应没有，f是干啥的？

**A：** F用于将Viewport相机聚焦到选中对象；先在Stage选择，再让Viewport获得键盘焦点。它是否生效不能单独判断模型是否加载。实际机器人位置和相机目标仍要对照检查。

**Q：** RANDOM也没机器人！草！我怎么改我的那个IMPORT脚本，一步一步排查？TITA_Lab/scripts/import_assets.py

**A：** 当时先做静态显示来隔离相机与模型加载问题。静态对照只调用render，不能检查重力、碰撞或执行器。当前脚本已经恢复run_simulator，不需要再次把正常物理循环改成静态循环；历史过程见页末原文。

**Q：** 有机器人。但是穿模，无物理效果，机器人就是原始USD的姿态，不动。

**A：** 那次现象对应临时的render循环，没有推进物理。当前入口已经调用run_simulator，循环中包含sim.step；应按当前代码和新运行现象判断，不再把当时的静态诊断视为现状。

**Q：** 有几个报错，不知道啥原因。这次也正常。

**A：** 那份历史日志显示执行器CUDA Graph捕获失败后退回逐步计算。回退不表示CPU执行，也不能据此判断全部物理数值。当前RL任务因自定义TitaPD选择Lab执行路径，独立资产脚本仍采用其现有配置；不把旧日志直接当作本轮必须改参数的证据。

## 历史原文入口

[早期静态显示、相机和执行器图捕获排查原文](../../../docs/history/2026-10-22_import_assets排查原文.md)。历史步骤不作为当前待办。
