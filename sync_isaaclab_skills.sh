#!/usr/bin/env bash
# 同步本地 Isaac Lab 技能为真实文件；不联网、不 git pull、不创建 symlink。
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ===== 可修改的配置：填写 IsaacLab 根目录和安装父目录（不是 skills 目录） =====
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/IsaacLab}"
INSTALL_DIR="${INSTALL_DIR:-$SCRIPT_DIR/.agents}"  # 最终安装到 "$INSTALL_DIR/skills/"
PYTHON_BIN="${PYTHON_BIN:-python3}"              # Python 3.10+，仅标准库
# =======================================================================

exec "$PYTHON_BIN" - "$ISAACLAB_ROOT" "$INSTALL_DIR" "$@" <<'PY'
"""Project-local, manifest-managed synchronization; no third-party dependencies."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote, unquote


def fail(message):
    raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def real_directory(path):
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        fail(f'目标必须是真实目录，不是链接或文件：{path}')


def files_under(folder, ancestors=()):
    """Dereference source symlinks, detect cycles, yield the logical copy paths."""
    resolved = folder.resolve(strict=True)
    if resolved in ancestors:
        fail(f'源码中存在循环目录链接：{folder}')
    for item in sorted(folder.iterdir()):
        if item.is_dir():
            yield from files_under(item, ancestors + (resolved,))
        elif item.is_file():
            yield item
        else:
            fail(f'源码包含无法复制的条目（例如失效链接）：{item}')


def inventory(folder):
    """Return content hashes; never traverse links in an existing destination."""
    if not folder.exists() and not folder.is_symlink():
        return None
    real_directory(folder)
    result = {}
    for path in folder.rglob('*'):
        if path.is_symlink():
            fail(f'安装目标中发现链接，请先处理该链接再同步：{path}')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = sha(path.read_bytes())
        elif not path.is_dir():
            fail(f'安装目标有不支持的文件类型：{path}')
    return result


def git_info(source):
    try:
        top = subprocess.check_output(['git', '-C', str(source), 'rev-parse', '--show-toplevel'],
                                      stderr=subprocess.DEVNULL, text=True).strip()
        if Path(top).resolve() != source:
            return None, None
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--', 'skills'],
                                            stderr=subprocess.DEVNULL, text=True).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def safe_relative(value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        fail(f'同步记录含有非法相对路径：{value}')
    return p


def main():
    if sys.version_info < (3, 10):
        fail('需要 Python 3.10+；可在脚本开头将 PYTHON_BIN 设为 /usr/bin/python3。')
    parser = argparse.ArgumentParser(
        prog='sync_isaaclab_skills.sh',
        description='将本地 Isaac Lab 技能同步为普通文件；保留无关技能，检查本地修改。',
        epilog='例：bash sync_isaaclab_skills.sh --dry-run；参数路径中有空格时加引号。')
    parser.add_argument('--source', default=sys.argv[1], help='IsaacLab 源码根目录（其中包含 skills/）')
    parser.add_argument('--target', default=sys.argv[2], help='安装父目录，通常为项目的 .agents；自动添加 skills/')
    parser.add_argument('--dry-run', action='store_true', help='只预览，不修改安装目录或创建备份')
    parser.add_argument('--force', action='store_true', help='允许覆盖本地修改；修改前自动备份')
    args = parser.parse_args(sys.argv[3:])
    source = Path(args.source).expanduser().resolve()
    target = Path(args.target).expanduser().absolute()
    if not source.is_dir() or not (source / 'skills').is_dir():
        fail(f'找不到源码 skills/：{source}；--source 应指向 IsaacLab 根目录。')
    if target.name == 'skills':
        fail('--target 应填 skills 的父目录（通常 xxx/.agents），不要填写 skills 本身。')
    # Do not write into the source checkout or through existing destination links.
    if target.resolve().is_relative_to(source) or source.is_relative_to(target.resolve()):
        fail('安装目录与 IsaacLab 源码目录不能互相包含；请选择独立项目的 .agents。')
    for parent in [target, *target.parents]:
        if parent.is_symlink():
            fail(f'安装路径包含目录链接：{parent}；请选择真实目录。')
    real_directory(target)
    skills_target = target / 'skills'
    real_directory(skills_target)
    manifest_path = target / 'isaaclab-skills-source.json'
    readme_path = target / 'ISAACLAB_SKILLS.md'
    license_path = target / 'ISAACLAB_SKILLS_LICENSE.txt'
    for p in (manifest_path, readme_path, license_path):
        if p.is_symlink() or (p.exists() and not p.is_file()):
            fail(f'元数据目标不是普通文件：{p}')
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    old_names = set(old.get('skills', {}))
    for name in old_names:
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', name):
            fail(f'同步记录技能名无效：{name}')
    old_hashes = {name: {} for name in old_names}
    for item in old.get('files', []):
        rel = safe_relative(item['file'])
        if len(rel.parts) < 2 or rel.parts[0] not in old_names:
            fail(f'同步记录不完整：{item["file"]}')
        old_hashes[rel.parts[0]][Path(*rel.parts[1:]).as_posix()] = item['installed_sha256']

    sources = {}
    for entry in sorted((source / 'skills').glob('*/*/SKILL.md')):
        match = re.search(r'^name:\s*([^\r\n]+)', entry.read_text(), re.M)
        name = match.group(1).strip().strip('\"\'') if match else ''
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', name) or name in sources:
            fail(f'缺少有效且唯一的 name：{entry}')
        sources[name] = entry.parent
    if not sources:
        fail('未找到 skills/<user 或 developer>/<目录>/SKILL.md；停止以免误删已安装技能。')
    commit, dirty = git_info(source)
    ref = commit or 'HEAD'
    print(f'源码：{source}\n安装：{skills_target}\n提交：{commit or "无法确定；外部引用使用 GitHub HEAD"}')
    print('仅同步本地文件，不执行 git pull。')
    if dirty:
        print('提示：本地 skills 有未提交改动；同步会采用本地内容，逐文件哈希将记录这些内容。')

    link_re = re.compile(r'(\]\()([^\s)]+)([^)]*\))')
    records, rewrites, desired = [], [], {}
    # Temporary preparation is outside the destination, including in --dry-run mode.
    with tempfile.TemporaryDirectory(prefix='isaaclab-skill-sync-') as temporary:
        prepared = Path(temporary)
        for name, folder in sources.items():
            desired[name] = {}
            for path in files_under(folder):
                relative = path.relative_to(folder)
                out = prepared / name / relative
                out.parent.mkdir(parents=True, exist_ok=True)
                raw = path.read_bytes()
                data = raw
                if path.suffix.lower() == '.md':
                    def replace(match):
                        value = match.group(2)
                        if value.startswith(('#', '/')) or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', value):
                            return match.group(0)
                        path_part, separator, fragment = value.partition('#')
                        resolved = (path.parent / unquote(path_part)).resolve()
                        if resolved.is_relative_to(folder.resolve()):
                            return match.group(0)
                        if not resolved.exists() or not resolved.is_relative_to(source):
                            fail(f'无效的跨目录技能引用：{path}: {value}')
                        new_value = None
                        for other_name, other_folder in sources.items():
                            if resolved.is_relative_to(other_folder.resolve()):
                                mapped = prepared / other_name / resolved.relative_to(other_folder.resolve())
                                new_value = Path(os.path.relpath(mapped, out.parent)).as_posix()
                                break
                        if new_value is None:
                            new_value = f'https://github.com/isaac-sim/IsaacLab/blob/{ref}/' + quote(
                                resolved.relative_to(source).as_posix(), safe='/')
                        if separator:
                            new_value += '#' + fragment
                        rewrites.append({'file': f'{name}/{relative.as_posix()}', 'from': value, 'to': new_value})
                        return match.group(1) + new_value + match.group(3)
                    data = link_re.sub(replace, raw.decode('utf-8')).encode('utf-8')
                out.write_bytes(data)
                out.chmod(path.stat().st_mode & 0o777)
                desired[name][relative.as_posix()] = sha(data)
                records.append({'file': f'{name}/{relative.as_posix()}',
                                'source': path.relative_to(source).as_posix(),
                                'source_sha256': sha(raw), 'installed_sha256': sha(data)})
        # Check every copied bundle and its relative links before touching the destination.
        for md in prepared.rglob('*.md'):
            for match in link_re.finditer(md.read_text()):
                value = match.group(2)
                if value.startswith(('#', '/')) or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', value):
                    continue
                if not (md.parent / unquote(value.split('#')[0])).exists():
                    fail(f'复制后有无效相对引用：{md}: {value}')

        actions, conflicts, before = {}, [], {}
        for name in sorted(set(sources) | old_names):
            current = inventory(skills_target / name)
            before[name] = current
            expected = desired.get(name)
            if current == expected:
                continue
            # Only directories named in the previous manifest can be removed.
            if name not in sources and current is None:
                continue
            if current is not None and (name not in old_names or current != old_hashes[name]):
                conflicts.append(name)
            actions[name] = 'REMOVE' if name not in sources else ('ADD' if current is None else 'UPDATE')
        for name, action in actions.items():
            print(f'{action:6} {name}')
        if conflicts:
            print('检测到本地修改或同名未托管目录：' + ', '.join(conflicts))
            if not args.force:
                print('未修改任何目标文件。保留/合并修改后重试，或使用 --force（会先备份）。')
                return 2
            print('--force：将先备份上述目录再同步。')

        metadata = {'format_version': 1, 'source_root': str(source), 'source_commit': commit,
                    'source_skills_dirty': dirty, 'method': 'project-local regular-file copy; no symlinks',
                    'skills': {name: folder.relative_to(source).as_posix() for name, folder in sources.items()},
                    'files': records, 'adapted_links': rewrites}
        old_comparable = {k: v for k, v in old.items() if k not in ('date', 'synced_at')}
        readme = f'''# Isaac Lab 技能 COPY 同步记录

由 sync_isaaclab_skills.sh 维护。来源：`{source}`；提交：`{commit or '未知'}`。

- 安装位置：`skills/`，共 {len(sources)} 个技能，全部为普通文件和真实目录。
- 技能内部支持文件随目录复制；跨目录引用适配为同项目技能路径或来源提交的 GitHub 链接。
- 上游的 docs/source、source、tools 等路径与命令指 Isaac Lab checkout。优先读匹配的本地源码，其他系统使用该机实际安装位置；没有匹配文件时再读链接。
- 本地源码更新后重新运行同步脚本；脚本不联网、不 git pull、不安装 Python 包或运行仿真。
- 脚本开头的 ISAACLAB_ROOT 是源码根目录，INSTALL_DIR 是安装父目录（通常 xxx/.agents）；也可用 --source 和 --target 临时覆盖。
- 默认检查本地修改；--dry-run 预览；--force 先备份再覆盖。备份在 skill-sync-backups/，无关技能不受影响。
- 详细来源、文件哈希和引用适配见 isaaclab-skills-source.json；许可见 ISAACLAB_SKILLS_LICENSE.txt。
'''
        license_source = source / 'LICENSE'
        if not license_source.is_file():
            fail(f'缺少源码许可文件：{license_source}')
        extras = {readme_path: readme.encode(), license_path: license_source.read_bytes()}
        changed_extras = {p: data for p, data in extras.items() if not p.exists() or p.read_bytes() != data}
        needs_metadata = metadata != old_comparable
        print(f'计划：{len(actions)} 个技能变更；{len(sources) - sum(n in sources for n in actions)} 个技能内容不变；'
              f'{len(rewrites)} 处引用适配；元数据更新={needs_metadata or bool(changed_extras)}')
        if args.dry_run:
            print('预览结束；安装目录没有改动。')
            return 0
        if not actions and not needs_metadata and not changed_extras:
            print('已是最新，无需修改。')
            return 0

        target.mkdir(parents=True, exist_ok=True)
        skills_target.mkdir(exist_ok=True)
        backup_root = target / 'skill-sync-backups'
        real_directory(backup_root)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        backup = backup_root / stamp
        backup.mkdir(parents=True)
        moved, installed, written = [], [], []
        # Same-filesystem staging enables directory renames and rollback on failure.
        with tempfile.TemporaryDirectory(prefix='.skill-sync-stage-', dir=target) as local_temp:
            local_stage = Path(local_temp)
            try:
                for name in actions:
                    if inventory(skills_target / name) != before[name]:
                        fail(f'同步期间目标被其他进程改动：{name}')
                    if name in sources:
                        shutil.copytree(prepared / name, local_stage / name)
                for name in actions:
                    path = skills_target / name
                    if path.exists():
                        saved = backup / 'skills' / name
                        saved.parent.mkdir(parents=True, exist_ok=True)
                        path.rename(saved)
                        moved.append(name)
                    if name in sources:
                        (local_stage / name).rename(path)
                        installed.append(name)
                        if inventory(path) != desired[name]:
                            fail(f'写入后内容校验失败：{name}')
                metadata['synced_at'] = datetime.now(timezone.utc).isoformat()
                changes = dict(changed_extras)
                # Write the manifest last so only a completed plan becomes the new baseline.
                changes[manifest_path] = (json.dumps(metadata, ensure_ascii=False, indent=2) + '\n').encode()
                for path, data in changes.items():
                    if path.exists():
                        shutil.copy2(path, backup / path.name)
                    written.append((path, path.exists()))
                    staged_file = local_stage / path.name
                    staged_file.write_bytes(data)
                    os.replace(staged_file, path)
            except BaseException:
                for path, existed in reversed(written):
                    if existed:
                        shutil.copy2(backup / path.name, path)
                    elif path.exists():
                        path.unlink()
                for name in reversed(installed):
                    shutil.rmtree(skills_target / name)
                for name in reversed(moved):
                    (backup / 'skills' / name).rename(skills_target / name)
                print('同步失败，已回滚本次文件变更。', file=sys.stderr)
                raise
        print(f'同步完成：{len(sources)} 个技能。备份：{backup}')
        return 0


try:
    sys.exit(main())
except (OSError, ValueError, KeyError, RuntimeError) as error:
    print(f'错误：{error}', file=sys.stderr)
    sys.exit(1)
PY
