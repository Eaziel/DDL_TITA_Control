# Sim6.2随机动作运行与FIR重复更新缺陷

日期：2026-10-27。内容纯AI生成，由人类要求生成，未经人类修改。

**本轮发现：当前process_actions重复执行两次“保存上一裁剪动作、写入当前裁剪动作”，使FIR退化为当前裁剪动作，同时action-rate原始惩罚恒为0。** 已给修正函数，用户尚未写入。它属于当前任务代码缺陷，不是Sim6.2接口移除。

## 用户运行结果

[日志片段](user_random_log_excerpt.txt)显示PhysX/cuda:0、4环境、seed42、五组观测与8动作，进入random-agent循环。末尾有`^C`后清理退出，片段无traceback；没有步数、推扰或超时计数，不能据墙钟运行约50 s断言完成22 s仿真预算。[结构化记录](runtime_summary.json)保留这个边界。

## 源码定位与影响

当前`TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/actions.py:53-60`重复两组copy_。第二次将previous_clipped_actions覆盖为当前c_t，于是：

$$
u_t=0.2c_t+0.8c_t=c_t.
$$

实际`mdp/rewards.py:72-84`还使用这两个缓存计算：

$$
r_{rate,raw}=\sum_j(c_{t,j}-c_{t-1,j})^2.
$$

缓存都等于当前动作后，该函数对每个环境均返回0。这里的0是原始惩罚，未乘奖励权重或dt。

## CPU复现与修正候选

执行[check_fir.py](check_fir.py)，解释器为新venv的Python；调用真实TitaJointAction方法、新Lab的JointAction父方法及真实action_rate_clipped_l2。手工创建CPU缓存，不构造机器人或物理环境。

内存候选只删除函数开头重复的两条copy_；中间英文字符串随即恢复为函数docstring。两组动作分别按六腿/两轮shape测试，四环境×12策略步，含±100裁剪。候选在独立模块构造；奖励函数的类型检查临时指向受测类，测试结束恢复，未修改源文件。

| 输入顺序 | 10 | 0 | 200 | -200 |
| --- | --- | --- | --- | --- |
| 当前实际滤波输出 | 10 | 0 | 100 | -100 |
| 正确0.2/0.8输出 | 8 | 2 | 80 | -60 |
| 修正候选输出 | 8 | 2 | 80 | -60 |

候选缩放/offset后的目标与公式最大差0；action-rate与八列直接求和的最大差0.015625，来自float32的六列+两列与八列求和顺序，在本测试的大数值下是舍入误差。当前版本所有action-rate均0。原始动作保存、输入不被修改的检查通过。[result.json](result.json)保存修正前方法原文、哈希与数值。

没有执行仿真、训练、Git或源码写入。此前其他日期的CPU通过只对应其当时受测内容，不能覆盖今天发现的重复更新。下一步由用户替换该方法，保留已有配置/导出，再做短随机动作运行；完整2200步、15s推扰、20s超时及传感器数值仍分别记录。

本轮按新checkout的isaaclab-using-sensors-actuators技能读取actuators概念文档、JointAction和ManagerBasedRLEnv实现。父类仍按scale/offset处理目标；process每策略步、apply每物理步的调用入口仍保留。旧Gym原式见`envs/legged_robot.py:643-645`，action-rate原式见同文件1189行。
