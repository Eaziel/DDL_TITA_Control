# observation_history.py：保存十帧自身观测

源码目录：`TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/`。
源码文件：[observation_history.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/observation_history.py)。

更新：2026-10-19。以下类及read已写入，history只读组也已登记；环境step推进也已写入。该普通Python缓存由TitaVelocityEnv持有，Env与MDP函数分别调用其写入和读取方法。真实环境尚未运行。

## 当前代码

```python
from collections.abc import Sequence

import torch


class ProprioHistory:
    def __init__(
        self,
        num_envs: int,
        obs_dim: int = 33,
        history_length: int = 10,
        device: str | torch.device = "cpu",
    ) -> None:

        if num_envs <= 0 or obs_dim <= 0 or history_length <= 0:
            raise ValueError("缓存三个维度必须为正整数.")

        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.history_length = history_length

        self._history = torch.zeros(
            (num_envs, history_length, obs_dim),
            dtype=torch.float32,
            device=device,
        )

    def read_then_push(
        self,
        current_obs: torch.Tensor,
        episode_length: torch.Tensor,
    ) -> torch.Tensor:

        if current_obs.shape != (self.num_envs, self.obs_dim):
            raise ValueError("current_obs 的形状不匹配.")
        if episode_length.shape != (self.num_envs,):
            raise ValueError("episode_length 的形状不匹配.")

        # 保存本次要返回的旧历史
        past = self._history.clone()

        # 普通步：丢掉最旧帧，追加当前帧
        shifted = torch.cat(
            (self._history[:, 1:, :], current_obs.unsqueeze(1)),  # unsq 加时间轴
            dim=1,
        )

        # 旧 Gym 的 开局 规则：用当前帧填满历史
        repeated_current = current_obs.unsqueeze(1).expand(
            -1,
            self.history_length,
            -1,  # (N, 1, 33)  --expand(-1, 10, -1)-->  (N, 10, 33)
        )
        self._history = torch.where(  # torch.where(条件, A, B)
            (episode_length <= 1)[:, None, None],  # 刚开局（第0、1步）。`[:, None, None]`：每环境一个 bool，广播用
            repeated_current,  # 刚开局，填满
            shifted,
        )

        # 先读取再push新的观测，这里返回的是push新观测前的
        return past

    # 历史缓存“只读取，不更新”。
    # Lab 可能为了确定观测维度、记录数据等原因，多次调用观测函数。
    # 读取一次观测，不应当等于时间前进了一步。因此，读取历史和写入新帧必须分开。
    def read(self) -> torch.Tensor:
        """Return a copy of the stored history without updating it."""

        return self._history.clone()

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:

        if env_ids is None:
            self._history.zero_()
        else:
            self._history[env_ids, :, :] = 0.0
```

## 对象保存什么

`ProprioHistory(...)` 创建一个缓存对象，环境把它保存在 `self.proprio_history`。缓存对象的 `_history` 是 `(N,10,33)` Torch张量，三个轴依次为环境、由旧到新的时间帧、当前帧33个特征。

类负责持有数据和组织读写，torch.zeros/cat/where等函数负责执行具体数组运算。定义类不会自己采集数据；必须有人创建对象，并在正确时机调用方法。

## 三个方法怎样配合

`read()` 返回 `_history.clone()`：得到独立副本，不更新。MDP观测函数调用它，随后展平成 `(N,330)`。额外读一次不会让历史多走一步。

`read_then_push(current_obs, episode_length)` 有两个结果：先复制旧历史为past；再更新对象内部的_history；最后返回此前的past。return虽然写在最后，返回的仍是更新前快照。

正常滑动的张量变化是：

```text
_history[:,1:,:]            (N,9,33)，丢最旧帧
current_obs.unsqueeze(1)   (N,1,33)，为当前帧加时间轴
沿dim=1拼接                 (N,10,33)，追加当前帧
```

开局时，`expand(-1,10,-1)` 把当前帧扩展成十份。`episode_length<=1` 为每个环境选回填或滑动；`[:,None,None]` 把条件变成 `(N,1,1)`，广播到时间与特征轴。

`reset(env_ids)` 只清指定环境的所有帧和特征；没有传env_ids才清全部。显式写 `_history[env_ids,:,:]`，环境轴的含义明确。

## 一次调用前后有什么区别

用3帧简化演示，真实工程为10帧：

| 状态 | 普通步 | 开局步 |
| --- | --- | --- |
| 原缓存 | `[10,11,12]` | `[0,0,0]` |
| 传入当前帧 | `13` | `23` |
| 返回past | `[10,11,12]` | `[0,0,0]` |
| 对象内部新缓存 | `[11,12,13]` | `[23,23,23]` |

数字代表整帧的标号。`read_then_push` 的调用方拿到旧快照；随后再调用read才读到新缓存。两者不指向同一份历史。

## 接入环境时由谁调用

当前方案由Env.step在进入父类step前，用已有上一份policy推进缓存；ObsTerm只通过read取数；Env._reset_idx清相应环境。Env.step中只使用read_then_push的更新效果，不接收其返回值。

当前read、只读组与Env.step均已写入，静态调用链完整；真实更新仍待运行检查。完整代码、Gym顺序与迁移理由见 [05第4—6节](../../../../../../../docs/migration/05_observations迁移.md#4-历史gym-原来怎么保存)。

此前保存代码与旧Gym真实torch.where更新语句做过独立CPU对照，检查了正常滑动、开局回填、部分重置和返回副本。详细数字与临时脚本位置保存在 [项目历史](../../../../../../../docs/PROJECT_HISTORY.md)。这些是局部张量检查，不是真实Lab接入或物理验证。

## 补充问答

**Q：**

> 还有reset,我记得obs hist也写了reset。有这么多reset，我没搞懂啊，我还记得reset是lab的方法，我们重写了是吗？是环境重置的时候，会自动调用所有被登记的reset？这个代码框架有点大，说实话我都忘了obs hist文件里头的内容拿来干啥了。

**A：** 此文件提供普通Python类ProprioHistory。环境用self.proprio_history=ProprioHistory(...)持有它；内部张量为(N,10,33)，保存每个环境过去十帧观测。step调用read_then_push推进窗口；观测函数调用read并展平为(N,330)，生成history组。当前普通PPO尚未消费history，后续迁移旧编码网络时再接入。

当前清理方法为：

```python
def reset(self, env_ids=None):
    if env_ids is None:
        self._history.zero_()
    else:
        self._history[env_ids, :, :] = 0.0
```

它是普通自定义方法，未重写Lab的方法。调用它的实际代码在TitaVelocityEnv._reset_idx：

```python
super()._reset_idx(env_ids)
self.proprio_history.reset(env_ids)
```

第二行把这份缓存接进回合重置；ObservationManager保存的是读取历史的函数，并不自动接管这个缓存对象。多个reset的对象归属、原生奖励类列表和入口链见[环境末尾问答](../tita_velocity_env.py.md#补充问答)。清历史是为避免新回合带入上一回合轨迹，选中环境以外的历史继续保留。
