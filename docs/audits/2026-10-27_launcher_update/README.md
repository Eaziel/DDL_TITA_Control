# Isaac Sim桌面/Dash启动器更新

2026-10-27。目标：桌面和Dash启动Sim6.2，关闭Sim后保留lab3_sim62交互终端，目录为~/IsaacLab-main。终端保留行为依据用户本轮追加要求。

## 已修改

- 桌面与用户应用目录的isaacsim.desktop沿用同一ID，名称改为Isaac Sim 6.2，并引用新环境图标；Dash无需重新固定。
- ~/.local/share/isaacsim-launcher/env.sh退出已激活的Conda、激活新venv并切目录；rc.sh加载终端设置后使用它，再运行isaacsim isaacsim.exp.full。Sim返回后继续原交互shell。
- 移除直接指向旧Conda的重复IsaacSim.desktop。
- ~/.bashrc中goisaac-6.1替换为goisaac-6.2，引用同一env.sh；没有删除旧Conda、Lab或修改其他shell初始化功能。

改动前文件保留在/home/eaziel/.local/share/isaacsim-launcher-backups/20261027_155007，含清单和哈希。这是普通启动配置备份，与项目Git备份无关。

## 验证

脚本bash -n和desktop-file-validate通过；桌面数据库刷新与gio设置可信均exit 0。Gio按原app ID读取到新入口，桌面/Dash文件一致，旧重复文件和旧alias已不存在。

对实际安装的rc使用返回7的模拟器替身：之后仍可执行命令，python与VIRTUAL_ENV指向新venv，CONDA_PREFIX为空，cwd为IsaacLab-main。没有启动真实Sim或观察GUI双击；真实图标运行由用户确认。详情见[result.json](result.json)。

用户同轮已提供TITA editable安装及list_envs成功日志；本轮元数据也确认TITA_Lab0.1.0、源码路径与isaaclab.tasks入口。注册通过不等于新版任务reset/step通过。

本轮改动已获用户明确授权；没有修改TITA_Lab、TITA_non_RL或操作Git。
