# 旧URDF与当前USD的碰撞定义

本页用途是保留该日期的检查范围与复现方法。文中的“当前/候选/待写”按检查当时理解；后续实现状态见[项目记忆](../../../PROJECT_MEMORY.md)。2026-10-26文档整理未重新执行本页检查。

2026-10-23。用户关闭自碰撞后，日志到第200策略步并正常清理；仅记录两次base_contact，分别62步、84步，两次other=0且terrain=total。见[用户日志节选](self_collision_disabled_excerpt.txt)。这强烈支持此前异常与自碰撞有关，但没有定位具体碰撞对。

首次只读XML/pxr快照见[result.json](result.json)，下表描述关闭四处碰撞前的差异；它是历史快照。最新静态结果见[after_user_usd_edit.json](after_user_usd_edit.json)：左右leg_1/leg_2四处Collision Enabled=False，其余五处=True。用户报告恢复自碰撞后现象正常，未附完整新日志。助手没有修改资产、源码或运行模拟器。

| 部位 | 旧Gym实际引用URDF中的有效collision定义 | 当前USD中的CollisionAPI几何 |
| --- | --- | --- |
| base_link | box，0.47×0.30×0.19 m，中心(0,0,0.017)，rpy=0 | Mesh，convexDecomposition |
| 左右leg_1 | 无 | 各一个Mesh，convexHull |
| 左右leg_2 | 无；collision块在XML注释中 | 各一个Mesh，convexHull |
| 左右leg_3 | box，0.20×0.07×0.05 m，中心(0.1,0.021,-0.002)，rpy=(0,0,-0.131) | 各一个Mesh，convexHull |
| 左右leg_4 | cylinder，半径0.0925 m、长度0.034 m，局部原点、局部z轴 | 各一个Mesh，convexHull |

计数是5个有效URDF collision元素与9个USD碰撞Mesh节点，不是已读取的PhysX cooked shape数量。凸分解可能生成多个凸形状。旧Gym选项replace_cylinder_with_capsule=False；助手未重新运行旧Gym获取其cooked形状。

当前机身源Mesh在body坐标系的包围尺寸约0.504163×0.312385×0.206750 m，中心(-0.006426,0,0.023025)。这与旧自定义box尺寸/位置不同，因此将当前Mesh下拉改Bounding Cube并不等价于旧URDF的box。该包围盒只是源Mesh范围，不是PhysX凸分解后形状或穿透测试。

九个刚体均显式写有质量、质心和对角主惯量。只禁用碰撞几何不等于删除这些刚体；后续改碰撞形状时也应保留质量参数，避免无意按新形状重新估算惯量。

## 当前结论与下一步

碰撞链接覆盖已修正：用户在源USD关闭四处额外碰撞并报告重新开启自碰撞后正常，不再要求补写此前Python覆盖规则。tita.py的enabled_self_collisions=True仍被注释，建议清理时改成显式True。当前机身/小腿仍为Mesh近似、轮子仍为凸包，完整几何对齐没有完成。

最新取舍：用户问“我不加行吗？”，建议本阶段跳过三个box及轮子圆柱替换，保留现有形状；清理诊断后转回地形配置。以下精确几何信息仅作可选对照。具体尺寸、局部pose与清理代码见[19](../../migration/19_碰撞模型对齐与诊断清理.md)。当前USD与旧URDF八条主关节零位变换最大元素误差约2.23e-8，支持直接采用旧link坐标；-0.131 rad对应-7.505747度。清理候选AST检查通过，未用户写入或运行；精确box/cylinder不作为继续迁移的前置条件。

碰撞几何决定接触位置、法线与接触力矩，进而影响平衡、轮地滚动、台阶通过及终止触发。优先复制原任务的简化碰撞定义建立迁移基线，再单独评估增加物理细节的影响。保留已有质量/COM/惯量与视觉，不把box collider当作重新估计惯量的依据。
