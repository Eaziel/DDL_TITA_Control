# Kit中机器人不可见：当前证据

本页用途是保留该日期的检查范围与复现方法。文中的“当前/候选/待写”按检查当时理解；后续实现状态见[项目记忆](../../../PROJECT_MEMORY.md)。2026-10-26文档整理未重新执行本页检查。

2026-10-22。用户zero_agent/单环境/Kit日志见[user_log.txt](user_log.txt)，只读检查摘要见[result.json](result.json)。窗口和地形可见，机器人不可见来自用户报告，助手未观察窗口；用户随后报告random_agent也不可见，未附该次日志。

零动作日志中八关节和cost(1,6)已建立，Ctrl+C后正常清理。本地USD有9个Mesh，未发现authoring visibility为invisible；这不能证明运行时已画出。当时相机固定看世界原点，粗糙地形生成10×20块且机器人出生偏移，是一条可验证的视角假设。日志另有Fabric同步警告，/Visuals/Command路径的警告对应命令箭头，不能直接认定为机器人网格错误。

原生KitVisualizerCfg跟随字段及default_visualizer_cfg传递在CPU核对过，未运行GUI。用户随后在import_assets.py写入实际位置相机与render-only循环，报告机器人可见，但原USD姿态、不动并穿模。只读核实当前main尚未调用run_simulator，故未执行其中的初值写入与物理循环；不能据此认定碰撞失效。当前未覆盖use_fabric，无关闭Fabric后恢复显示的证据。

用户现已恢复run_simulator，报告“这次也正常”；只读核实当前main调用已恢复。本轮只附[四行CUDA日志](user_dynamic_excerpt.txt)，没有完整运动/接触数据。日志对应原生执行器CUDA Graph捕获失败后eager回退；本地physx_runtime.py明确实现该回退。独立脚本默认use_newton_actuators=True，RL配置为False。先前建议在独立脚本关闭原生执行器路径属于旁支，不作为不可见排查的前置步骤。

## 已取得用户运行确认：跟随相机后机器人可见

2026-10-22用户按下述方案修改RL相机后报告“好了”，能看到机器人；只读核实env_cfg已采用Kit asset跟随，TitaVelocityEnv.step已加首步位置打印。用户没有贴坐标输出，助手未观察GUI。这确认跟随相机解决了本次不可见，不扩展为物理或训练已验证。

2026-10-22用户纠正排查跑偏后，按当前源码追查：

1. env_cfg.py:683只有VisualizerCfg(eye=(8,0,5))。lookat默认世界原点，Kit的origin_type默认world。
2. ROUGH_TERRAINS_CFG每块8米，10行20列。TerrainImporter._compute_env_origins_curriculum中，num_envs=1令terrain_types[0]=floor(0/(1/20))=0；初始行从0到5抽取。函数名包含curriculum并不表示本任务已开启课程。
3. TerrainGenerator将全图中心移到世界原点。第0列中心y=(0+0.5)×8−20×8/2=−76米；前六行的x为−36、−28、−20、−12、−4、4米。
4. reset_root_state_uniform用positions=default_root_pose[:,:3]+env.scene.env_origins+rand_samples[:,:3]生成实际位置。当前默认xy为0，重置xy各±1米，因此初始机器人y约−76米，而非世界原点。

以上坐标来自配置和原生源码推导，未取得用户的实际root_pos_w输出；跟随后可见来自用户运行报告。独立脚本正常只证明同一资产在独立场景可显示/运行。

用户已在RL环境用KitVisualizerCfg(origin_type='asset', origin_env_index=0, origin_track_path='robot', eye=(1.5,1.5,1), lookat=(0,0,0))替换固定世界视角，使用sim.visualizer_cfgs列表。原生KitVisualizer读取scene['robot'].data.root_pos_w.torch[0]，将其分别加到eye/lookat上，所以相机对准真实机器人位置。首策略步同时打印env_origin与root_pos_w，让预测有实际数值依据。

配置实例在当前解释器构造及validate通过，位置公式的标准Python算术检查通过；助手未启动GUI。完整参考改动和数据流解释见[env_cfg说明](../../../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/config/Tita/env_cfg.py.md)。两个源码工程均未由助手修改。

## 旧Gym的实际相机行为与下一步

旧envs/base_task.py:97–99默认free_cam=False、lookat_id=0、lookat_vec=(0,2,1)。lookat(i)读取root_states[i,:3]，相机位置为该位置加lookat_vec，目标为该机器人位置；render在free_cam=False时每帧调用。viewer.pos/lookat只覆盖初始化及切入自由相机时的固定视角，不能用它们概括正常跟随行为。Lab现在的asset跟随对应这一职责，若需相同初始方向可将eye改为(0,2,1)。此为界面视角，不参与策略动作、物理终止或奖励计算。

用户最新担忧刚落地就频繁reset。下一步转到[07终止检查](../../migration/07_terminations迁移.md)，记录重置前回合步数/仿真秒数及具体终止项；尚未取得原因日志，不能判断正常或异常。
