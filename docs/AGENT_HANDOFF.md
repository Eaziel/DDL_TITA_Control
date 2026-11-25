# Agent接手说明：任务、代码职责与接续方法

更新：2026-11-14。内容纯AI生成，由人类要求生成，未经人类修改。

本页帮助新助手建立项目结构和技术边界。先读[项目记忆入口](../PROJECT_MEMORY.md)和必读的[协作与讲解](../project_memory/协作与讲解.md)；迁移进度统一维护在[迁移状态](../project_memory/迁移状态.md)，权限以[AGENTS](../AGENTS.md)为准。

## 1. 先确认正在完成的任务

主任务是把DDT TITA的Isaac Gym任务及其学习计算接入IsaacLab-main／Sim6.2，最终获得可持续训练、保存加载和评估的TITA策略。用户希望理解每个模块为什么存在、数据怎样进入训练；源码由用户自己写。目录组织采用Lab的职责划分，不逐文件复制旧仓库。

本项目同时有研究、职业和内容材料。它们的提案不自动成为RL迁移要求。当前任务只依据用户最新消息与项目记忆判断，不能从旁支材料补出一个新课题。

## 学习背景与讲解示例

本部分已合并到必读的[协作与讲解](../project_memory/协作与讲解.md)：包含用户明确的学习基础、怎样判断眼前困惑、局部修改怎样给出位置、docstring与中文注释示例，以及回复和详细文档各自的作用。开始工作时完整阅读该文，不根据已手敲或已运行推断用户已经掌握。

这里保留阅读入口，避免在接手说明里重复维护另一份讲解方式。

## 2. 接手时最少读哪些文件

1. [AGENTS](../AGENTS.md)：源码、Git、文档和运行权限。
2. [PROJECT_MEMORY](../PROJECT_MEMORY.md)与必读的[协作与讲解](../project_memory/协作与讲解.md)：先理解怎样协作，再按任务选主题。
3. 继续迁移时读[迁移状态](../project_memory/迁移状态.md)；涉及路径、版本或物理验证时再读[运行环境与验证边界](../project_memory/运行环境与验证边界.md)；研究话题读[独立专题](../project_memory/独立专题.md)。
4. 需要代码结构时读本页后续调用链。定位模块用[重要代码解释索引](重要代码解释索引.md)，查参数来源用[00映射表](migration/00_迁移映射索引.md)，核对运行结论再打开相关[audit](audits/README.md)。

不默认读取全部历史聊天、迁移章节和上游仓库。框架API按当前本地源码核实；项目[技能副本](../.agents/ISAACLAB_SKILLS.md)按任务选择，不为接手自动同步、安装或升级。

## 3. 两条训练路线必须看清

### 已注册的官方PPO模板

`tasks/velocity/config/Tita/__init__.py`登记任务及`rsl_rl_ppo_cfg.py:PPORunnerCfg`。模板仍是小型PPO配置：16步／轮、最大150轮，只读policy组33列，experiment_name仍为cartpole_direct。模板没有自动引用我们自己的Gaussian policy、critic、六cost和storage。

因此，任务能列出rsl_rl不表示自有NP3O已经接入官方train入口。模板以后如何处置要随真实训练接线决定，本轮不修改它。

### 当前自有check

源码根目录：`../Tita_Lab/`。按实际调用关系读：

```text
scripts/check_np3o_interface.py: main()
  resolve_task_config(... agent_cfg_entry_point=None, physics=isaacsim_physx)
  launch_simulation → gym.make → Gymnasium外层 → TitaVelocityEnv
  TitaNP3OEnvWrapper(gym_env) → run_check(wrapper, max_steps)
    新建policy、critic和storage → reset
    采集T次：旧obs下采样动作和预测价值 → env.step → 保存经历
      环境：动作处理 → 每策略步4次物理推进 → reward/cost/结束标记/观测
      wrapper：五组观测打包586列，附六cost和结束信息
    取末尾价值 → compute_returns（GAE）
    mini_batch_generator：检查同一经历各字段的取数
    check_value_update：固定目标MSE，critic的Adam更新及拟合图
    check_policy_update：奖励PPO与六类cost惩罚，policy的Adam更新
    退出；没有下一轮采集，没有保存训练checkpoint
```

`env`名称不代表一种新环境类型：wrapper.env保存gym.make的返回对象，wrapper._env直接指向其unwrapped底层TitaVelocityEnv；check中的lab_env也指向该底层对象。调用后看到多个变量名，不等于创建了多个物理世界。详细说明见[Wrapper入门](训练流程与Wrapper入门.md)。

check使用的`learning/np3o/`当前只有actor、policy、critic、env_wrapper、rollout_storage及初始化文件。没有自有algorithm／runner。check中的两个局部Adam是真实参数更新；奖励与六类cost单批策略更新均已有用户运行，仍没有构成持续训练流程。

## 4. 当前各文件负责什么

| 位置（相对Tita_Lab/） | 主要职责 | 接续时常见误判 |
| --- | --- | --- |
| src/Tita_Lab/assets/tita.py | USD、名义初值与执行器 | GUI显示不能代替碰撞和动力学验证 |
| tasks/velocity/config/Tita/env_cfg.py（包内） | 选场景、action、command、obs、event、reward、termination、curriculum及物理参数 | cfg选择实现；跨步缓存和计算由具体类负责 |
| tasks/velocity/mdp/（包内） | 单项状态计算、控制、事件与有状态项 | rewards.py／costs.py计算单步量；critic预测累计价值 |
| tasks/velocity/tita_velocity_env.py（包内） | 将历史、cost和奖励等扩展接入Lab step/reset | 父类与自定义代码共同决定时序 |
| tasks/velocity/observation_layout.py（包内） | 按固定顺序检查并拼接586列 | 不可随意换列，模型用固定切片 |
| tasks/velocity/cost_manager.py（包内） | 六项cost、dt缩放、缓存、回合统计及k/d | 提供cost不等于约束已经影响策略梯度 |
| learning/np3o/env_wrapper.py（包内） | 整理环境接口、传递成本与最终观测等信息 | wrapper不实现学习算法 |
| learning/np3o/actor.py、policy.py（包内） | 历史动作均值网络；高斯分布、采样与密度 | actor辅助方法存在，不表示训练已调用 |
| learning/np3o/critic.py（包内） | 从当前状态及编码特征预测奖励／六cost价值 | 有价值预测不表示预测准确或任务学会 |
| learning/np3o/rollout_storage.py（包内） | 复制经历、GAE、小批量 | step是本批写入计数，num_steps是本批容量 |
| scripts/check_np3o_interface.py | 一批数据上的接口、梯度、参数更新与绘图检查 | 每次重新初始化；图不是完整训练学习曲线 |

表中“包内”统一指`Tita_Lab/src/Tita_Lab/`，不是项目根目录的同名文件夹。详细可点击路径见[重要代码索引](重要代码解释索引.md)。

## 5. 不应在接续中破坏的数据约定

以下来自既有源码核对与证据。本轮定向核实check、wrapper、注册和PPO模板；没有重跑所有物理／张量计算。改相关模块前仍须读其当前源码。

| 内容 | 当前约定 |
| --- | --- |
| 官方路线与自有检查 | 默认登记rsl_rl/PPO模板，num_steps_per_env=16、max_iterations=150，只读policy33。自有check不加载agent cfg，直接创建policy/critic/storage；其24来自CLI。官方入口不会自动使用这些自有模块。 |
| 时间 | PhysX；物理dt=.0025s，decimation=4，策略步=.01s；任务回合20s，check可单次覆盖回合长度。 |
| 动作顺序 | 左腿1/2/3、右腿1/2/3、左轮、右轮，共8列。 |
| 控制 | 裁剪→两点FIR→缩放/默认角→物理步目标历史→延迟选择→随机PD因子→限矩。sim.use_newton_actuators=False。 |
| 延迟 | 共享(N,)编号；腿/轮缓冲(N,3,6)/(N,3,2)。编号0/1/2对应5/2.5/0ms，startup采一次；reset恢复目标历史。 |
| 打包观测 | policy33、scan187、privileged_state6、dynamics30、history330，依此拼586列并裁到±100；辅助目标220:230，历史从256开始。 |
| 观测历史与噪声 | 过去10帧，不含当前；step开头推进，getter只读，reset清选中行。noise→clip→scale；四项最终半宽.2/.05/.01/.075，轮角置零后仍可加噪声。 |
| 高度与额外状态 | scan offset=.4、clip=(-1,1)、scale=5，射线抬高20m不在输出再减20。六维为机身线速度×2、左右轮接触−.5、延迟编号/(L−1)。接触按Fz>1N，与前策略步原始判断OR，每策略步只采一次。 |
| 材质与动力学 | startup前保存名义质量/COM，再读实际材质与电机因子；详细规则见events与observations解释。 |
| 命令与奖励 | base_velocity=(N,3)，[vx,vy,wz]；航向目标另存。功率是带符号机械功率；加速度按.01s差分。普通奖励合计夹零后再加失败项。 |
| cost | 六列顺序acc_smoothness/dof_vel_limits/feet_contact_forces/pos_limit/stumble/torque_limit。原始项不乘dt/k；cost管理器只乘一次策略dt；k=[.1,.3,.1,.3,.1,.3]，d全0。 |
| 模型 | actor用9帧旧历史+当前构成330→26，与当前33合59→8均值；辅助支路用原样过去10帧。policy直接学习8个std，初值1。critic共享编码，输出奖励1列、cost6列。critic与含六类cost惩罚的策略各完成8次检查性更新；辅助学习未接入。 |
| 存储 | (T,N,D)复制13项来源，另有四份GAE结果。num_steps是单批容量，step是批内已写数量；先保存旧obs对应经历，再obs=next_obs。clear归零计数并将_returns_ready设False，不由机器人reset自动调用；compute_returns正常结束设True。 |
| GAE | γ=.99、λ=.95，storage保存原始优势；check策略更新时按整批统计量标准化reward优势，cost优势已逐类接入约束策略损失。普通步取下一旧价值，纯超时取reset前最终价值，真实终止取0；所有done处停止优势跨回合递推。 |
| 最终与末尾价值 | cfg.compute_final_obs=True使Lab在reset前捕获final_obs。wrapper转发terminated；check选择超时且未真实终止的行算timeout_*。last_*是采集结束后最后obs的价值，两者不可混用。 |
| 生命周期 | .pyi参与MDP lazy_export；RewardManager调用登记奖励类的reset，历史和cost由环境显式接线。wrapper读取已保存cost，不重复计算/乘dt。 |


586列切片明确为：policy`0:33`、scan`33:220`、privileged_state`220:226`、dynamics`226:256`、history`256:586`。N是并行环境数，T是每环境本批策略步数，B是一次小批量记录数。当前N=4、T=24产生96条记录；4份给B=24，2轮给每个优化器8次更新。每环境仿真时间T×.01=.24s，不把并行环境的时间相加当作单机器人经历。

## 6. 单批约束更新完成后，怎样判断下一步

2026-11-14用户已完成六类cost进入同一次策略更新的短检查，[证据](audits/2026-11-14_constrained_policy_update/README.md)包含非零合计cost梯度与参数变化。当前按用户要求先学习[check讲义](../code_explaination/Tita_Lab/scripts/check_np3o_interface.py.md)，第12节从基本导数关系融入autograd；不自动进入runner。

持续训练前仍需处理初始累计cost估计、优化器跨批状态、辅助学习与熵开关、配置及保存加载。当前J估计依赖从reset开始的首步GAE目标，不能直接移入任意批次起点。d目前全0，非零阈值须先明确单位。具体当前状态以[迁移状态](../project_memory/迁移状态.md)为准。

独立算法来源记录仍见[30](migration/30_NP3O来源与接入边界.md)。OmniSafe固定版本为计算对照，不是已安装的六cost N-P3O依赖；本轮没有clone、安装或复用厂商完整算法。

## 7. 对证据和解释的要求

- 接口shape／有限值／参数改变，分别只支持对应层次的结论；任务性能需要更新后回放和任务指标。
- 实际图／CSV优先于“还没生成”的旧待办。本次图的分析见[11月14日记录](audits/2026-11-14_critic_plot/README.md)。图在critic阶段保存，不凭图断言该次policy和最终退出完成。
- 学习基础和解释写法按[协作与讲解](../project_memory/协作与讲解.md)落实，明确反馈更新在该页；不从已敲代码或通过检查推断熟练度。
- 运行证据按各记录的日期、输入和执行范围使用。本次学习交接整理只更新普通文档，没有运行Torch、Sim、训练、安装或Git。
