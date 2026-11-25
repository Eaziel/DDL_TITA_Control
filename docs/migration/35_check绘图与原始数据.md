# 35 check中的critic拟合图与原始数据

内容纯AI生成，由人类要求生成，未经人类修改。更新：2026-11-14。

**状态：已经生成实际PNG／PDF、两份CSV及metadata。** 用户运行产物位于Tita_Lab/logs/interface_checks/20261113_212733_632325；本轮读取并分析了全部五个文件。详见[实际图、数值与证据边界](../audits/2026-11-14_critic_plot/README.md)。缺少run_metadata的调用已由用户补齐；旧补丁不重复应用。

## 1. 保存了什么

- measure_value_fit：更新前及每次critic更新后，对固定T×N条观测和GAE目标重新计算MSE；eval／no_grad，随后恢复原模式。
- save_value_plots：保存奖励价值MSE、六cost平均MSE、奖励价值更新前后散点，以及原始CSV与运行元数据。
- 保留原小批量loss统计和更新检查；绘图不改变损失、学习率或更新次数。
- 本次N=4、T=24，共96条记录；critic更新8次，连同更新前测量共有9个点。metadata记录seed42、策略dt=.01s、20s回合、γ=.99、λ=.95、Adam3e-4、4份×2轮和cost顺序。

输出目录每次独立新建。critic_fit.png为300dpi图，critic_fit.pdf用于排版；value_metrics.csv含全部cost分列MSE，reward_predictions.csv含每条奖励目标和前后预测，metadata.json说明计算设置。

## 2. 本次读图结果

| 整批指标 | 更新前 | 更新后 | 相对下降 |
| --- | --- | --- | --- |
| 奖励价值MSE | 0.315601 | 0.123848 | 60.76% |
| 六cost平均MSE | 0.952881 | 0.631117 | 33.77% |

平均cost误差约94.8%由acc_smoothness贡献；stumble首尾仅改善0.81%。奖励总体改善主要来自env3，另外三个环境的MSE上升。详细分项和分环境表在上述分析中，不能仅凭两条总曲线说每项或每条经历都改善。

## 3. 这些图回答的是什么

每次曲线测量使用相同整批数据和固定GAE目标。原CLI loss是更新过程中不同小批量loss的平均，与图的统计口径不同。散点横轴为目标、纵轴为预测，y=x代表该条预测等于目标。

这是单批、单seed的样本内拟合诊断：GAE目标含旧价值自举，尚非精确真实价值；96条经历互相关联，四个环境也不等于四次独立seed实验。当前没有更新后任务回报、跟踪性能或约束下降的结果。图在critic阶段保存，不能只凭图补出同次后续policy和退出日志。

## 4. 接线历史与详细解释

用户曾遇到check_value_update缺少run_metadata。保存源码现已补齐，实际图证明保存链已执行；该问题不再是待修项。[旧补丁](../patches/2026-11-13_check_value_plots.patch)只留作变更格式参考，不重复套用。

[check解释第14节](../../code_explaination/Tita_Lab/scripts/check_np3o_interface.py.md)讲评估、CSV、元数据与图；第16节讲.patch文件和GNU patch命令。旧候选AST／dry-run只属于当时静态证据；本次产物来自用户运行，助手没有代运行模型、仿真或绘图，也没有改源码或Git。
