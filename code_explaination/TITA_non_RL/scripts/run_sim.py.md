# run_sim.py 代码解释

解释的文件目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_non_RL/scripts/`。

解释的文件：[run_sim.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_non_RL/scripts/run_sim.py)。

原解释日期：2026-10-15；文档整理：2026-10-18。本页按“源码→创建随机桶→按环境取值→广播材质→写入后端”的顺序保留原解释。它对应非RL独立脚本，当前RL环境的随机化状态见项目记忆。本次未修改脚本或运行仿真。

## 摩擦与恢复系数随机化

### 源码

```python
# 官方方式：摩擦和恢复系数分别准备 64 个随机值
mu_buckets = torch.empty(64, device="cpu").uniform_(0.2, 2.75)
e_buckets = torch.empty(64, device="cpu").uniform_(0.0, 1.0)

# 每个环境分别选取一个摩擦值、一个恢复系数
n = robot.num_instances
mu = mu_buckets[torch.randint(64, (n,), device="cpu")]
e = e_buckets[torch.randint(64, (n,), device="cpu")]

# 同一环境内，整个机器人的碰撞体共用这组参数
view = robot.root_view
materials = wp.to_torch(view.get_material_properties()).clone()
materials[..., 0] = mu[:, None]  # 静摩擦
materials[..., 1] = mu[:, None]  # 动摩擦，与静摩擦相同
materials[..., 2] = e[:, None]   # 恢复系数

env_ids = torch.arange(n, dtype=torch.int32, device="cpu")
view.set_material_properties(
    wp.from_torch(materials, dtype=wp.float32),
    wp.from_torch(env_ids, dtype=wp.int32),
)

print("每个环境的摩擦和恢复系数：", torch.stack((mu, e), dim=-1))
```

### 解释（对话原文）

可以。这段代码从编程上分为：**创建数组 → 用索引取值 → 修改三维数组 → 传给另一个库的接口。**

下面用 `n = 3`，即三个环境举例。

**1. 创建两个长度为 64 的张量**

```python
mu_buckets = torch.empty(64, device="cpu").uniform_(0.2, 2.75)
e_buckets = torch.empty(64, device="cpu").uniform_(0.0, 1.0)
```

拆开第一句：

```python
mu_buckets = torch.empty(64, device="cpu")
mu_buckets.uniform_(0.2, 2.75)
```

- `torch.empty(64)`：创建形状为 `(64,)` 的一维张量。刚创建时，里面的数值没有初始化。
- `device="cpu"`：把数据放在 CPU 内存。
- `.uniform_(0.2, 2.75)`：用均匀分布的随机数填充这个张量。
- 方法名末尾的 **`_` 表示原地修改**，直接改已有张量里的数据。

`uniform_()` 还会返回这个张量，所以可以接在 `torch.empty()` 后面连续调用。

最后得到类似：

```python
mu_buckets = tensor([0.7, 1.8, 0.4, ..., 2.1])  # 共 64 个数
```

**2. 创建随机整数索引，再按索引取数**

```python
n = robot.num_instances
mu = mu_buckets[torch.randint(64, (n,), device="cpu")]
```

这里比较绕，可以拆成：

```python
n = robot.num_instances
indices = torch.randint(64, (n,), device="cpu")
mu = mu_buckets[indices]
```

`torch.randint()` 的参数含义是：

```python
torch.randint(
    64,             # 整数取值：0～63，包含 0，不包含 64
    (n,),           # 输出形状：长度为 n 的一维张量
    device="cpu",
)
```

**`(n,)` 是 Python 的单元素元组。** 末尾逗号让它成为元组；`(n)` 只是加了括号的 `n`。

例如：

```python
indices = tensor([2, 2, 7])
mu = mu_buckets[indices]
```

相当于取出：

```python
mu = tensor([
    mu_buckets[2],
    mu_buckets[2],
    mu_buckets[7],
])
```

这叫**使用整数张量进行索引**。索引可以重复，所以不同环境可能取到相同数值。

下面这句完全同理，只是从另一个张量里取：

```python
e = e_buckets[torch.randint(64, (n,), device="cpu")]
```

此时 `mu`、`e` 的形状都是 `(n,)`。

**3. 取得数据，并复制成一个可修改的张量**

```python
view = robot.root_view
materials = wp.to_torch(view.get_material_properties()).clone()
```

拆开是：

```python
view = robot.root_view                  # 取得一个后端视图对象
warp_array = view.get_material_properties()  # 调用方法，取得 Warp 数组
torch_view = wp.to_torch(warp_array)     # 用 PyTorch 张量访问它
materials = torch_view.clone()          # 复制出独立的数据
```

这里的 `view` 是变量名，不是调用 PyTorch 的 `.view()` 方法。

`materials` 是三维张量，形状为：

```text
(n, 碰撞形状数量, 3)
```

假设三个环境，每个机器人有九个碰撞形状，就是：

```text
(3, 9, 3)
```

最后一维的三个位置依次存放静摩擦、动摩擦、恢复系数。

**4. 使用切片和广播批量赋值**

```python
materials[..., 0] = mu[:, None]
materials[..., 1] = mu[:, None]
materials[..., 2] = e[:, None]
```

先看左边：

```python
materials[..., 0]
```

这里 `...` 表示“前面的维度全部保留”。对这个三维张量，它等价于：

```python
materials[:, :, 0]
```

也就是：取所有环境、所有形状的第 0 列。结果形状为 `(n, 9)`。

再看右边：

```python
mu[:, None]
```

- `:`：保留 `mu` 中的全部元素。
- `None`：插入一个长度为 1 的新维度。

例如：

```python
mu = tensor([0.4, 1.2, 2.0])       # 形状：(3,)

mu[:, None]
# tensor([[0.4],
#         [1.2],
#         [2.0]])                  # 形状：(3, 1)
```

赋值时：

```text
左边：(3, 9)
右边：(3, 1)
```

PyTorch 会将右边每一行的一个值自动扩展到九个位置，这叫**广播（broadcasting）**。

所以第一行最终相当于：

```text
环境 0：[0.4, 0.4, …, 0.4]
环境 1：[1.2, 1.2, …, 1.2]
环境 2：[2.0, 2.0, …, 2.0]
```

**5. 将张量交给后端接口**

```python
env_ids = torch.arange(n, dtype=torch.int32, device="cpu")
```

`torch.arange(n)` 生成从 `0` 到 `n-1` 的整数：

```python
tensor([0, 1, 2], dtype=torch.int32)
```

接下来：

```python
view.set_material_properties(
    wp.from_torch(materials, dtype=wp.float32),
    wp.from_torch(env_ids, dtype=wp.int32),
)
```

这是普通的 Python 方法调用，传入两个参数：

1. 修改后的数据。
2. 这些数据对应的环境编号。

`wp.from_torch()` 把 PyTorch 张量包装成 Warp 数组，供这个接口使用；这里指定的数据类型与张量本身一致，不是用它做数值类型转换。

最后：

```python
torch.stack((mu, e), dim=-1)
```

将两个形状为 `(n,)` 的张量沿新维度拼起来，得到 `(n, 2)`：

```text
[[mu₀, e₀],
 [mu₁, e₁],
 [mu₂, e₂]]
```

因此打印出来，每行就是一个环境的一组参数。
