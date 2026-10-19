# 项目代码解释

**内容纯 AI 生成，由人类要求生成，未经人类修改。** 本声明针对本目录的说明文档。

本目录按源码文件归档Python结构与数据流解释。每页先给当前源码与顺序解释，用户追问放在后面的Q/A；要求保留的旧原文单独标明历史状态。Gym计算、Lab代码与迁移理由集中在[迁移说明](../docs/migration/README.md)，无需为拼出一个功能逐份查本目录。当前[观测历史接线](../docs/migration/05_observations迁移.md#5-历史lab-中怎样接入)的完整接线已写入并静态核对，尚未运行真实环境。

说明归档规则（2026-10-15，按用户要求）：

- `TITA_non_RL/` 对应项目根目录下的 `TITA_non_RL/`。
- `TITA_Lab/` 对应项目根目录下的 `TITA_Lab/`。
- 各子目录内镜像源码的相对目录；说明文件名使用原源码文件名加 `.md`，例如 `scripts/run_sim.py.md`。
- 同一个源码文件的解释合并维护在同一个 Markdown 文件中；不同源码文件分别建文件，不按对话或日期拆散。
- 每份说明开头注明所解释源码的目录和文件路径；每个解释先贴出对应源码，再给解释。
- 本次归档的讲解原文保持不变。后续新解释追加到对应文档，源码变化时同步维护相关片段与状态，避免把旧示例当成当前实现。
- 补充问答使用Q：、A：，Q保留用户原句；用户明确表示不知怎样提问时才整理并注明。主线必需的定义与计算仍随代码顺序解释。
- 当前正文不插入失去问题背景的澄清或逐轮修正；历史原文保持原样。
- 明确区分源码检查、示例和实际运行验证。

## 已有说明

- [TITA_non_RL：scripts/run_sim.py](TITA_non_RL/scripts/run_sim.py.md)
- [TITA_Lab 归档入口](TITA_Lab/README.md)
- [TITA_Lab：assets/tita.py](TITA_Lab/assets/tita.py.md)
- [TITA_Lab：scripts/import_assets.py](TITA_Lab/scripts/import_assets.py.md)
- [TITA_Lab：env_cfg.py 当前配置块与历史解释](TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/env_cfg.py.md)
- [TITA_Lab：mdp/observations.py 的轮角屏蔽函数](TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/observations.py.md)
- [TITA_Lab：mdp/observation_history.py 的10帧缓存与Torch API（已写入，完整接线已写入）](TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/observation_history.py.md)
- [TITA_Lab：历史的构造、step推进与重置](TITA_Lab/src/TITA_Lab/tasks/velocity/tita_velocity_env.py.md)
- [TITA_Lab：已修改的环境类入口与原配置入口的区别](TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/__init__.py.md)
- [TITA_Lab：mdp/__init__.pyi 的自定义观测导出](TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/__init__.pyi.md)
- [TITA_Lab：mdp/terminations.py 当前机身接触终止（已写入并局部检查）](TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/terminations.py.md)
- [TITA_Lab：mdp/rewards.py 当前帧小腿碰撞计数（已保存并局部核对）](TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/rewards.py.md)
- [TITA_Lab：mdp/events.py 水平速度推扰（待写方案）](TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/events.py.md)
