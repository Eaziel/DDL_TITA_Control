# 2026-11-04：rollout与check短采集记录

内容纯AI生成，由人类要求生成，未经人类修改。

## 用户提供的运行证据

[user_check_excerpt.txt](user_check_excerpt.txt)保存本轮用户粘贴的输出片段，未收到完整日志文件。片段显示：

- 四个环境，24/24次并行策略步；storage写入计数为24/24。
- observations=(24,4,586)，actions=(24,4,8)，costs=(24,4,6)，共96条环境交互。
- 单步策略/价值输出维度符合当前接口，代码中的断言未中止本次运行。
- 1次回合结束、0次超时，随后SimulationContext清理，应用正常退出。

当前源码的策略周期为0.01s，因此24步对应每环境累计0.24s；片段中的13.871s是应用墙钟时间，不能当作单环境仿真时间。reset原因没有逐次记录，尾部没有完整启动命令、设备或种子行。

## 助手只读核对

核对源码：

- `Tita_Lab/src/Tita_Lab/learning/np3o/rollout_storage.py`。
- `Tita_Lab/scripts/check_np3o_interface.py`。
- policy、wrapper与观测打包函数的当前返回及数据所有权。

storage已经存在，check已经导入和调用；来源shape与11项存储约定一致。旧obs在env.step后仍保留，先复制旧obs、动作及结果，再采用next_obs。用户保留超时键判断后再累计的写法有效，未发现本步必须修改的源码问题。

仅进行文本、AST与文档核对；未代执行Torch模块、模型、项目脚本、仿真、训练或任何Git操作。用户的运行证据与助手的静态检查分开记录。

## 结果边界与阅读入口

本次验证采集与保存链，不证明价值预测准确或NP3O训练完成。源码没有计算回报/优势、调用backward或优化器更新，也没有将rollout写入磁盘。0次超时无法验证超时目标处理，短运行不覆盖15s推扰或20s回合上限。

- [storage解释](../../../code_explaination/Tita_Lab/src/Tita_Lab/learning/np3o/rollout_storage.py.md)：先理解保存数据供谁使用，再读(T,N,D)、unsqueeze和独立复制。
- [check解释](../../../code_explaination/Tita_Lab/scripts/check_np3o_interface.py.md)：沿main与run_check逐步追踪模型、环境和storage调用。
- [29阶段记录](../../migration/29_训练数据存储.md)：迁移范围与来源。
