# 旧生成项目注册到新版共享venv：核查结果

日期：2026-10-27。目标：判断已有TITA_Lab能否沿用新main的项目发现机制，并确定不重新解析旧依赖的安装命令。

## 结论

无需重建TITA项目。旧/new模板都使用uv_build、src包布局和isaaclab.tasks入口；四份包初始化/扫描/任务注册模板逐字相同。新版增加Stubbed内容和AMP等支持，任务模块合并为Jinja模板；发布版依赖选择由精确版本改为兼容版本，源码项目仍记录checkout路径。依据为本地两个checkout的tools/template及新版官方生成器文档，不读取Git。

当前旧isaaclab.sh本身已转交isaaclab.cli，-n最终进入tools/template/cli.py。新版官方入口为uv run isaaclab --new；用户已激活指定venv后可直接运行isaaclab --new。当前TITA是已有工程，不执行生成器。

## 安装方案与实际检查

目标解释器：/home/eaziel/my_venvs/lab3_sim62/.venv/bin/python。
TITA元数据、README/LICENSE、src/TITA_Lab存在；注册入口为TITA_Lab.tasks。新环境尚无TITA_Lab发行包/isaaclab.tasks入口。工作区同名目录可能被识别为namespace，不能把find_spec非空单独当作安装成功。

已执行uv --no-config --offline --no-cache pip install --dry-run --no-deps --no-sources，显式指定上述Python及TITA工程绝对路径。结果exit 0，Resolved 1 package，计划仅添加tita-lab。只读沙箱无法创建环境锁临时文件，uv报告锁警告；没有实际构建或安装，不证明构建依赖获取、完整任务导入或仿真成功。

推荐用户执行：

```bash
source "$HOME/my_venvs/lab3_sim62/.venv/bin/activate"
uv --no-config pip install \
  --python "$HOME/my_venvs/lab3_sim62/.venv/bin/python" \
  --no-deps --no-sources \
  -e /mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab
```

-e建立当前源码的editable安装；--no-deps不重算旧依赖；--no-config/--no-sources避免沿用旧uv配置/来源。基础依赖由新Lab环境提供。本步是已有环境中的任务包注册，不是把TITA的旧pyproject转换为新版独立环境清单；之后仍不在TITA目录uv sync。

安装后先打印TITA_Lab.__file__，应落在src/TITA_Lab/__init__.py，再运行isaaclab list_envs --keyword TitaLab。任务列出与真实reset/step分别验证。

## 已定位的后续适配

1. mdp/events.py仍用physics_manager.__name__；新main持有backend_manager实例，需改实例类型判断。旧预检与本轮源码核实一致。
2. tests/test_registration.py还断言isaaclab.envs:ManagerBasedRLEnv；实际任务注册已为自定义TitaVelocityEnv，应更新测试预期。这是项目迁移后测试未同步，不是新版不支持注册。
3. 旧pyproject中的Lab sources/Torch等覆盖尚未修改；采用上述注册命令临时绕开旧环境同步配置，正式独立同步适配另行处理。

没有修改两个源码工程、venv或Git，没有启动生成器、仿真或训练。source_read_snapshot.json仅记录本轮读取的代码/配置哈希，不代表提交或完整环境备份。

来源：[官方生成器](https://isaac-sim.github.io/IsaacLab/main/source/developer-tools/template_generator.html)、[uv editable安装](https://docs.astral.sh/uv/pip/packages/)。
