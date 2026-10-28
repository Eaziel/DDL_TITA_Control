# 课程配置当前源码复核

本页用途是保留该日期的检查范围与复现方法。文中的“当前/候选/待写”按检查当时理解；后续实现状态见[项目记忆](../../../PROJECT_MEMORY.md)。2026-10-26文档整理未重新执行本页检查。

2026-10-25。结果见[result.json](result.json)。三文件AST通过；GYM台阶选择、.pyi官方函数导出、CurriculumCfg条目、总配置字段及生成器布尔开关均已保存。仅检查当前源码，未启动Sim或训练。

本轮解释区分配置阶段与运行阶段：VelocityEnvCfg.__post_init__修改配置对象，设置生成器curriculum布尔值；真实CurriculumManager.compute则在reset前调用已登记的课程函数，更新地形等级和出生原点。compute是普通方法名，各管理器实现、返回值及副作用不同。

给用户下一步为PhysX/cuda:0、Kit、4环境2200零动作策略步（22s仿真时间）的有界运行。命令已核对本地CLI的zero_agent、max_steps和viz别名，助手未执行。零动作回合可能提前结束，因此运行总时长超过20s不代表某个回合触发过time_out；实际以用户新日志为准。旧CPU课程数值证据仍见[2026-10-23对照](../2026-10-23_terrain_curriculum/README.md)。
