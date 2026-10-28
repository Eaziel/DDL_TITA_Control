# 当前地形检查与课程候选

本页用途是保留该日期的检查范围与复现方法。文中的“当前/候选/待写”按检查当时理解；后续实现状态见[项目记忆](../../../PROJECT_MEMORY.md)。2026-10-26文档整理未重新执行本页检查。

2026-10-23。证据见[result.json](result.json)。只读用户源码、CPU夹具检查；未修改TITA_Lab或运行仿真/训练。

用户已保存tasks/velocity/terrain_cfg.py，两套实际名字为LAB_TITA_ROUGH_TERRAINS_CFG和GYM_TITA_ROUGH_TERRAINS_CFG；均validate通过，Lab预设副本及其sub_terrains独立。env_cfg导入路径有效，当前仍选Lab混合地形，尚未选TITA台阶；课程尚未登记。

候选只做：切到GYM方案、在.pyi导出官方terrain_levels_vel、添加CurriculumCfg和字段、同步地图curriculum开关。两个候选文件AST通过；不新增自定义课程函数，也不改变环境step/reset。

比较从旧源码AST直接提取_update_terrain_curriculum，init_done=True；使用原生CurriculumManager和TerrainImporter.update_env_origins，在相同torch随机种子下比较等级及出生原点。升级/降级/保持/未选环境、严格阈值/最低级、最高级随机回退、零命令零距离、空选择四组检查通过；课程日志键Curriculum/terrain_levels正确。

物理位置和命令是CPU替身；没有实例化真实场景、运行自动reset或训练。工具Warp报告无可见CUDA设备，但CPU检查正常exit 0。旧Gym构造结束时init_done已为True，不能把注释解读成每次首次公开reset必跳过课程；本方案直接复用官方函数，没有另加counter==0判断。完整迁移说明见[20](../../migration/20_地形课程迁移.md)。
