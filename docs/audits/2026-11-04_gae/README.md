# 2026-11-04：GAE普通与含超时的两次短检查

内容纯AI生成，由人类要求生成，未经人类修改。

## 用户提供的证据

[普通运行](user_normal_excerpt.txt)与[短回合运行](user_timeout_excerpt.txt)来自本轮粘贴的输出片段，未收到完整日志与启动参数。两次均为4环境24/24步，filled=24/24，各96条交互，四份GAE结果shape符合(T,N,1)/(T,N,6)，有限值检查未中止，正常退出。

普通运行统计1次reset、0次超时；短回合运行统计16次reset、16次超时。timeouts是报告时间上限标记的次数；原片段未逐步打印terminated与bootstrap_timeouts，所以不将16直接当作已逐项核验的纯超时补偿次数。

## 当前源码只读检查

storage已增加两个超时价值输入和四份结果，compute_returns使用完整批次、旧价值、末尾价值及GAE倒序递推。check启用compute_final_obs，按超时且未真实终止选择最终观测，调用critic后填回原环境行；采集结束后计算last_values并调用compute_returns。wrapper已转发terminated。未发现本步必须修改的源码问题。

此结论来自文本/AST核对与用户运行摘录；助手没有运行Torch、仿真、训练或Git。没有替用户修改源码。之前8组标准Python数学样例为独立计算边界核对，不能冒充当前Torch逐元素对照。

本轮文档核对：三份源码哈希未变，三份完整源码快照逐字对应；65个Python片段按所在语法上下文解析、68个本地链接及公式定界检查通过。三步教学数值与9组“GAE加权多步估计=TD误差加权和”的恒等式用标准Python核对，未导入Torch或项目模块。

## 结果边界

两次短运行说明采集与GAE计算链能够完成，未出现相应断言错误。输出只打印shape而非逐项数值，不能单凭它证明所有数值精度或旧Gym逐数一致。网络仍随机初始化，没有损失、backward或优化器更新，也没有完整训练runner或学习后策略评估。

## 阅读入口

- [storage解释](../../../code_explaination/Tita_Lab/src/Tita_Lab/learning/np3o/rollout_storage.py.md)：第13节从TD目标、多步估计到GAE与方法逐句说明。
- [check解释](../../../code_explaination/Tita_Lab/scripts/check_np3o_interface.py.md)：第7节追踪每步超时数据，第9节说明整批末尾与GAE调用。
- [31迁移记录](../../migration/31_回报目标与GAE.md)。
