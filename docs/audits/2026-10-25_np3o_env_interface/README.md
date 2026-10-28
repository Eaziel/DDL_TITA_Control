# NP3O环境接口候选检查

本页用途是保留该日期的检查范围与复现方法。文中的“当前/候选/待写”按检查当时理解；后续实现状态见[项目记忆](../../../PROJECT_MEMORY.md)。2026-10-26文档整理未重新执行本页检查。

2026-10-25。内容纯AI生成，由人类要求生成，未经人类修改。

候选来源：proposal_env_wrapper.py。当前用户拼接函数：TITA_Lab/src/TITA_Lab/tasks/velocity/observation_layout.py，按实际文件加载。未修改两个源码工程，未验收上一章commands候选。

运行：

```bash
/home/eaziel/miniconda3/envs/env_isaacSL-6.1.0/bin/python -B docs/audits/2026-10-25_np3o_env_interface/check_interface.py
```

result.json记录哈希与检查范围。四环境CPU替身验证：586各区间及裁剪不改源张量；构造不reset；reset/get读取；六项step返回；cost取自extras且不缩放；动作不裁剪/重排；done和两种时域的time_outs；log在done时转episode；不污染原生extras；回合步数转发及close。替身特意把当前cost管理器缓冲设零而在extras返回非零结束步cost，防止接口错读来源。

为避免启动仿真，检查脚本绕过tasks包注册导入；环境、传感器和reset物理均为替身。没有执行原生PhysX、整套旧runner、梯度更新或checkpoint转换。原生RSL包装器仅作源码依据，未在此实例化。当前Conda Python无Ruff模块，Ruff未执行。

候选仍需用户写入后核实；完整说明见[迁移22章](../../migration/22_训练环境接口.md)。
