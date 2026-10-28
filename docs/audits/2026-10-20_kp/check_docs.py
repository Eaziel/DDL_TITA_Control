"""Validate the maintained documentation without reading Git or unfinished actions."""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT=Path('/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR')
BEFORE=Path('/tmp/tita_audit_20261020/documents_before')
OUT=ROOT/'docs/audits/2026-10-20_kp/documentation_check.json'
files=[ROOT/'README.md',ROOT/'AGENTS.md',ROOT/'PROJECT_MEMORY.md',
       *sorted((ROOT/'docs').rglob('*.md')),*sorted((ROOT/'code_explaination').rglob('*.md'))]

def prefix(s):return s.split('## 历史解释原文',1)[0]
def is_archive(p):return 'docs/history/' in str(p.relative_to(ROOT))
def questions(s):
    return [re.sub(r'\s+',' ',x.strip()) for x in re.findall(r'\*\*Q：\*\*(.*?)\*\*A：\*\*',prefix(s),re.S)]
def headings(s):
    result=set();fence=False;counts=collections.Counter()
    for line in prefix(s).splitlines():
        if line.lstrip().startswith('```'):fence=not fence;continue
        if fence:continue
        if not re.match(r'^#{1,6} ',line):continue
        name=re.sub(r'^#+\s+','',line).strip().lower()
        name=re.sub(r'[^\w\-\s\u4e00-\u9fff]','',name)
        name=re.sub(r'\s','-',name)
        count=counts[name];counts[name]+=1
        result.add(name+(f'-{count}' if count else ''))
    return result

report={'files':len(files),'current_lines':0,'fence_errors':[],'broken_links':[],
        'broken_anchors':[],'lost_questions':[],'verbatim_history':{},'migration_lines':{},
        'assistant_source_writes':False,'git_accessed':False}
for p in files:
    s=p.read_text()
    if sum(line.lstrip().startswith('```') for line in s.splitlines())%2:
        report['fence_errors'].append(str(p.relative_to(ROOT)))
    if is_archive(p):continue
    report['current_lines']+=prefix(s).count('\n')
    if p.parent==ROOT/'docs/migration':report['migration_lines'][p.name]=s.count('\n')
    content=prefix(s)
    # File/API links in historical raw sections are intentionally not retargeted.
    for target in re.findall(r'\]\(([^)]+)\)',content):
        if target.startswith(('https://','http://','mailto:','codex:','app:')):continue
        target=target.strip('<>')
        filename,sep,anchor=target.partition('#')
        filename=unquote(filename)
        filename=re.sub(r':\d+$','',filename)
        q=(p.parent/filename).resolve() if filename else p
        if not q.exists():
            report['broken_links'].append({'file':str(p.relative_to(ROOT)),'target':target})
        elif anchor and q.suffix=='.md' and not is_archive(q) if q.is_relative_to(ROOT) else False:
            if unquote(anchor) not in headings(q.read_text()):
                report['broken_anchors'].append({'file':str(p.relative_to(ROOT)),'target':target})
    old=BEFORE/p.relative_to(ROOT)
    if old.exists():
        actual=collections.Counter(questions(s))
        # Asset page was moved to mirror the actual source path.
        if p==ROOT/'code_explaination/TITA_Lab/assets/tita.py.md':
            actual=collections.Counter(questions((ROOT/'code_explaination/TITA_Lab/src/TITA_Lab/assets/tita.py.md').read_text()))
        if p==ROOT/'docs/PROJECT_HISTORY.md':
            actual=collections.Counter(questions((ROOT/'docs/history/2026-10-20_整理前记录.md').read_text()))
        for question,count in collections.Counter(questions(old.read_text())).items():
            if actual[question]<count:report['lost_questions'].append({'file':str(p.relative_to(ROOT)),'question':question})

for rel in ['code_explaination/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/env_cfg.py.md',
            'code_explaination/TITA_Lab/src/TITA_Lab/tasks/velocity/tita_velocity_env.py.md']:
    old=(BEFORE/rel).read_text();new=(ROOT/rel).read_text()
    report['verbatim_history'][rel]=old[old.index('## 历史解释原文'):]==new[new.index('## 历史解释原文'):]
old=(BEFORE/'code_explaination/TITA_Lab/assets/tita.py.md').read_text()
new=(ROOT/'code_explaination/TITA_Lab/src/TITA_Lab/assets/tita.py.md').read_text()
report['verbatim_history']['assets/tita.py.md']=old[old.index('## 历史解释原文'):]==new[new.index('## 历史解释原文'):]
report['raw_project_history_preserved']=(BEFORE/'docs/PROJECT_HISTORY.md').read_bytes()==(ROOT/'docs/history/2026-10-20_整理前记录.md').read_bytes()
report['stage_result']=json.loads((OUT.parent/'result.json').read_text())['failures']
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
