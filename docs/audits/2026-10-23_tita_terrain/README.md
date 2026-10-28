# TITA原生台阶配置候选与当前清理检查

本页用途是保留该日期的检查范围与复现方法。文中的“当前/候选/待写”按检查当时理解；后续实现状态见[项目记忆](../../../PROJECT_MEMORY.md)。2026-10-26文档整理未重新执行本页检查。

2026-10-23。结果见[result.json](result.json)。用户源码与USD未修改；没有启动Sim、PhysX或训练。

当前三文件AST通过；重置诊断与COST打印已删，自碰撞显式True，正式历史/奖励/cost调用保留。剩余清理为Scene.base_contact_debug、step首步VIEW打印和cost临时断言。

候选使用原生Hf正/负金字塔台阶，10×20块、8m、外围25m、0.1m/0.005m离散、阶宽0.61m、平台参数3m、|h|=0.025+0.09d。负地形在字典前，curriculum布局使前十列向下、后十列向上。真实10×20配置validate通过；没有生成整张10×20地图。

六个原生CPU样本在d=0/0.5/0.9下网格和origin有限、正负高度方向正确；原生2×2 CPU生成器完成，origin shape=(2,2,3)、两列类型正确。工具进程Warp初始化报告无可见GPU，但CPU流程正常exit 0；不能将其当作用户GPU运行证据。

原生行内难度扰动、单块边缘填充和origin查询造成的差异保留；没有本机Isaac Gym二进制作逐点比较。本阶段用原生API迁移地形种类、主要参数和排列，自动升降级随后接CurriculumCfg。

候选完整代码和解释：[terrain_cfg.py说明](../../../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/terrain_cfg.py.md)。场景主线见[02](../../migration/02_scene迁移.md)。

后续目录决定：用户询问共用地形配置的位置后，建议从config/TITA上移到tasks/velocity/terrain_cfg.py；result.json中的旧拟建路径保留为原检查时快照，当前组织以项目记忆和说明为准。该调整未移动任何用户源码。
