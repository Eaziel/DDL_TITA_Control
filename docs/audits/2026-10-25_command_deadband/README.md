# 平移速度命令死区候选检查

本页用途是保留该日期的检查范围与复现方法。文中的“当前/候选/待写”按检查当时理解；后续实现状态见[项目记忆](../../../PROJECT_MEMORY.md)。2026-10-26文档整理未重新执行本页检查。

2026-10-25。候选源码：[proposal_commands.py](proposal_commands.py)；结果：[result.json](result.json)。只在docs保存候选，未修改用户TITA_Lab源码，未启动Sim或训练。

候选继承原生UniformVelocityCommand并覆盖_resample_command，先执行父类，再按sqrt(vx²+vy²)>0.2保留平移命令。配置类指向新实现，沿用父类时间、航向、静止概率和marker设置。

检查使用原生生成器构造函数、reset、compute及CPU机器人/可视化注册表替身；参考直接提取旧_resample_commands最后一条归零语句AST。六组定值样本覆盖零、正负方向、边界及逐分量小于阈值但合速度超过阈值的情况；128环境的全部与部分reset通过，未选行保持。连续1002策略步中父子计时器/命令计数一致，本次原生重采样发生在第1000步；航向目标和转向命令相同，命令shape仍(128,3)，cfg.validate通过。

本次只补目标数值规则，不对齐旧Gym与Lab的随机数消耗序列，也不改变奖励/命令回调位置。工具Warp初始化提示无可见CUDA设备，CPU检查正常exit 0。上述是原候选的检查范围。后续已核实用户写入实现、.pyi及cfg，当前实际配置导入、validate与class_type身份断言通过，见[current_wiring.json](current_wiring.json)；未运行新仿真。说明见[21](../../migration/21_命令死区迁移.md)。
