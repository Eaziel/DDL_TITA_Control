# 2026-11-14：六类cost接入单批策略更新

来源：用户在当前聊天中运行后粘贴的完整输出，原样保存在[user_run.txt](user_run.txt)。助手只读核对源码、CSV和metadata，没有代运行Torch、仿真或优化器。

## 本次闭环

任务TitaLab-Velocity-Tita，4环境、24步、seed42、PhysX，策略步长0.01s，gamma=0.99、lambda=0.95。命令为lab3_sim62环境中的python scripts/check_np3o_interface.py --num_envs 4 --max_steps 24 --device cuda:0 --viz kit。

采集96条经历，4份小批量重复2轮。critic与含六类cost惩罚的策略分别完成8次更新，末尾CHECK为24/24步、1次reset、0次超时，SimulationContext清理并正常关闭。

| 策略统计 | 用户输出 | 含义 |
| --- | --- | --- |
| 总损失过程平均 | 0.034586 | 奖励损失与cost惩罚之和 |
| 奖励损失过程平均 | -0.035403 | 标准化优势的PPO裁剪损失 |
| cost惩罚过程平均 | 0.069989 | 六类正违反量按k加权求和 |
| 首次密度比均值 | 1.000000 | 源码同时检查逐元素接近1 |
| 首次合计cost梯度L2范数 | 0.2595 | 合计cost损失对策略参数确有非零梯度 |
| 参数最大变化 | 0.002395 | 至少一个策略参数改变 |

六类代理违反量的过程平均依次为0.699890、-0.698050、-0.675381、-0.739003、-0.689720、-0.713550。顺序为acc_smoothness、dof_vel_limits、feet_contact_forces、pos_limit、stumble、torque_limit。惩罚在显示精度上几乎等于0.1×第一列平均值，说明本轮合计惩罚主要来自加速度项；平均值不能单独证明其余各列在每个小批量都未激活。

只读确认[本次绘图目录](../../../Tita_Lab/logs/interface_checks/20261114_181101_818344/)存在CSV、PNG、PDF和metadata。value_metrics有9点，reward MSE 0.315601→0.123848，cost平均MSE 0.952881→0.631117。它们在critic阶段保存，与后续策略loss不是同一种指标。

## 结论限度

本次完成“六类cost约束目标接入一批策略更新，并取得运行证据”的小目标。初始累计cost来自旧critic参与的首步GAE目标，仍为接线检查用近似；六列代理违反量不是实际物理违反次数。合计梯度非零不证明六类逐项都有非零梯度，也不证明更新后真实约束减少。

没有持续采集、保存加载、训练后回放或约束满足结果；辅助损失与熵项仍未接入。后续继续以[迁移状态](../../../project_memory/迁移状态.md)为准，不自动推进新模块。
