#!/usr/bin/env python3
"""Build a bilingual research report and local reference library with Python Markdown."""
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit
import markdown
from markdown.extensions.toc import slugify_unicode

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'_site'
ANALYSIS=ROOT/'artifacts/analysis'
REPO='https://github.com/XBQNRCM/ShopSimRL/blob/master/'
CHAPTERS=[('method','Method','方法'),('protocol','Experimental protocol','实验协议'),
          ('results','Results & skill evolution','结果与技能演化'),('behavior','Behavior analysis','行为分析'),
          ('optimization','Optimization & systems','优化与系统'),('reproduce','Reproduce & extend','复现与扩展')]
FILES={}

def emit(path,content):
    target=OUTPUT/path;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(content,encoding='utf-8');FILES[path]=target

def copy(source,path):
    target=OUTPUT/path;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target);FILES[path]=target

def relative(path,current):return os.path.relpath(path,Path(current).parent).replace('\\','/')

def table(headers,rows):
    return '\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'

def substitutions(lang):
    b=json.loads((ANALYSIS/'behavior/behavior.json').read_text(encoding='utf-8'))
    w=json.loads((ANALYSIS/'wandb/summary.json').read_text(encoding='utf-8'))
    zh=lang=='zh';result={};rows=[]
    for r in b['rounds']:
        for o in ['success','failure']:
            d=r[o];m=d['metrics'];rows.append([f"R{r['round']}",('成功' if o=='success' else '失败') if zh else o,
                d['n'],*[f"{m[k]['mean']:.2f}" for k in ['model_steps','tokens_per_turn','searches','unique_products']]])
    result['behavior_rounds']=table(['轮次','结果','n','模型步','token/步','搜索','不同商品'] if zh else ['Round','Outcome','n','Turns','Tokens/turn','Searches','Unique products'],rows)
    names=[('model_steps','Model turns','模型步'),('tokens_per_turn','Tokens / turn','token / 步'),('generated_tokens','Tokens / episode','token / episode'),('searches','Searches','搜索'),('unique_products','Unique products','不同商品'),('pagination','Pagination','翻页'),('invalid_actions','Invalid actions','无效动作'),('protocol_errors','Protocol errors','协议错误')]
    rows=[]
    for k,en,cn in names:
        m=b['paired']['fixed_val_all']['metrics'][k];lo,hi=m['ci95']
        rows.append([cn if zh else en,f"{m['a_mean']:.3f}",f"{m['b_mean']:.3f}",f"{m['difference']:+.3f}",f'[{lo:+.3f}, {hi:+.3f}]'])
    result['behavior_pairs']=table(['指标','Base','Iter80','差值','95% 区间'] if zh else ['Metric','Base','Iter80','Difference','95% interval'],rows)
    labels={'largest_step_reduction_both_success':('Both successful: largest step reduction','两端均成功：步数下降最多'),
            'failure_to_success_largest_token_reduction':('Failure → success: largest token reduction','失败 → 成功：token 降幅最大'),
            'success_to_failure_longest_final':('Success → failure: longest final trajectory','成功 → 失败：最终轨迹最长')}
    cases=[]
    for case in b['cases']:
        title=labels[case['selection_rule']][int(zh)]
        parts=[f'<details><summary>Task {case["task_id"]} · {title}</summary>']
        for cohort,d in case['conditions'].items():
            m=d['metrics'];label='Base' if cohort=='val_base' else 'Iter80'
            parts.append(f'<p><strong>{label}</strong> · {int(m["model_steps"])} turns · {int(m["generated_tokens"]):,} tokens · success={int(m["success"])}</p>')
            parts.append('<div class="case-path">'+html.escape(' → '.join(d['actions']))+'</div>')
        cases.append(''.join(parts)+'</details>')
    result['behavior_cases']='\n'.join(cases)
    result['wandb_runs']=table(['Round','Run','State','History events'],[[f"R{r['round']}",f"[{r['id']}]({r['url']})",r['state'],r['history_rows']] for r in w['runs']])
    keys=['train/entropy_loss','train/pg_clipfrac','train/grad_norm','perf/actor_train_tok_per_s']
    result['wandb_rounds']=table(['Round','Entropy','Clip fraction','Gradient norm','Actor tokens/s'],[[f"R{r['round']}",*[f"{r['metrics'][k]['mean']:.5f}" if j<2 else f"{r['metrics'][k]['mean']:.2f}" for j,k in enumerate(keys)]] for r in w['rounds']])
    return result

def render(text,prefix=''):
    maths=[]
    def stash(m):
        key=f'MATHPLACEHOLDER{len(maths)}END';tag='div' if m[0].startswith('\\[') else 'span'
        maths.append(f'<{tag} class="math">{html.escape(m[0])}</{tag}>');return key
    parts=re.split(r'(```.*?```)',text,flags=re.S)
    text=''.join(part if part.startswith('```') else re.sub(r'\\\[.*?\\\]|\\\(.*?\\\)',stash,part,flags=re.S) for part in parts)
    content=markdown.markdown(text,extensions=['fenced_code','tables','toc'],extension_configs={'toc':{'slugify':slugify_unicode}})
    for i,value in enumerate(maths):
        key=f'MATHPLACEHOLDER{i}END';content=content.replace('<p>'+key+'</p>',value).replace(key,value)
    content=re.sub(r'<table>(.*?)</table>',r'<div class="table-scroll"><table>\1</table></div>',content,flags=re.S)
    if prefix:
        content=re.sub(r'id="([^"]+)"',lambda m:f'id="{prefix}{m[1]}"',content)
        content=re.sub(r'href="#([^"]+)"',lambda m:f'href="#{prefix}{m[1]}"',content)
    return content

def shell(current,title,body,active='',source=None):
    rel=lambda p:relative(p,current)
    nav=''.join(f'<a href="{rel("chapters/"+slug+".html")}" '+('aria-current="page" ' if active==slug else '')+f'data-en="{en}" data-zh="{zh}">{en}</a>' for slug,en,zh in CHAPTERS)
    meta='<div class="report-meta">IntraSkill / ShopSimRL · Qwen3.5-4B · Iter80 · 4 rounds</div>'
    if source:meta+=f'<div class="source-note"><span data-en="Original source document; content retains its source language." data-zh="原始资料，正文保留源文档语言。">Original source document; content retains its source language.</span> <a href="{rel("sources/"+source)}" download>Markdown ↓</a></div>'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} | IntraSkill</title><link rel="icon" href="{rel('favicon.svg')}"><link rel="stylesheet" href="{rel('style.css')}"><link rel="stylesheet" href="{rel('report.css')}"><script src="{rel('reader.js')}" defer></script><script async src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"></script></head><body>
<a class="skip" href="#main" data-en="Skip to content" data-zh="跳至正文">Skip to content</a><header><nav class="wrap"><a class="brand" href="{rel('index.html')}">Intra<span>Skill</span><small> / ShopSimRL</small></a><div class="navlinks"><a href="{rel('index.html')}" data-en="Overview" data-zh="全览">Overview</a><a href="{rel('library.html')}" data-en="Reference library" data-zh="资料库">Reference library</a></div><button id="language" class="lang" type="button">中文</button></nav></header>
<div class="report-shell"><aside class="report-sidebar"><strong data-en="RESEARCH REPORT" data-zh="研究报告">RESEARCH REPORT</strong><div class="chapter-nav">{nav}<a href="{rel('library.html')}" data-en="Source documents" data-zh="原始文档">Source documents</a></div><div id="outline"></div></aside><main id="main" class="report-main">{meta}{body}<div class="chapter-next"><a href="{rel('index.html')}" data-en="← Project overview" data-zh="← 项目全览">← Project overview</a><a href="{rel('library.html')}" data-en="Reference library →" data-zh="资料库 →">Reference library →</a></div></main></div><footer><div class="wrap"><p>IntraSkill / ShopSimRL · Four completed rounds · Frozen evidence, reproducible analysis.</p></div></footer></body></html>'''

def main():
    d=json.loads((ANALYSIS/'analysis.json').read_text(encoding='utf-8'))
    assert d['primary_steps']==[1,80] and len(d['training_steps'])==80
    for name in ['index.html','style.css','app.js','favicon.svg','report.css','reader.js']:
        copy(ROOT/'site'/name,name)
    copy(ROOT/'ShopSimulator/assets/logo.png','assets/shopsimulator-logo.png')
    for p in sorted((ANALYSIS/'figures').glob('*')):
        if p.suffix in ['.svg','.png']:copy(p,'figures/'+p.name)
    for name in ['analysis.json','training_steps.csv','training_groups.csv','rounds.csv','chunk_contributions.csv','test_task_outcomes.json','provenance.json']:
        copy(ANALYSIS/name,'data/'+name)
    for folder,names in [('behavior',['behavior.json','extraction.json','provenance.json','train_episodes.csv.gz','train_turns.csv.gz','eval_episodes.csv.gz','eval_turns.csv.gz']),
                         ('wandb',['manifest.json','summary.json','rollout_steps.csv','optimizer_updates.csv','8ov14yyc.json','chjizc8u.json','7vhbv5j0.json','y4xk6byf.json'])]:
        for name in names:copy(ANALYSIS/folder/name,'data/'+folder+'/'+name)
    subs={lang:substitutions(lang) for lang in ['en','zh']}
    for slug,en,zh in CHAPTERS:
        body=''
        for lang in ['en','zh']:
            text=(ROOT/f'site/chapters/{slug}.{lang}.md').read_text(encoding='utf-8')
            for key,value in subs[lang].items():text=text.replace('{{'+key+'}}',value)
            assert '{{' not in text
            body+=f'<article data-language="{lang}" lang="{lang}"'+(' hidden' if lang=='zh' else '')+'>'+render(text,'zh-' if lang=='zh' else '')+'</article>'
        emit('chapters/'+slug+'.html',shell('chapters/'+slug+'.html',en,body,slug))
    sources=list((ROOT/'docs').rglob('*.md'))+[ROOT/'README.md',ROOT/'README.zh-CN.md',ROOT/'ShopSimulator/README.md']
    sources+=list((ROOT/'ShopSimulator/docs').rglob('*.md'))+list((ROOT/'artifacts').rglob('README.md'))
    sources+=[ROOT/'artifacts/cold-start/selected_skill.md']+list((ROOT/'artifacts').glob('round-00[0-3]/gate/selected_skill.md'))
    mapping={p.resolve():'reference/'+p.relative_to(ROOT).with_suffix('.html').as_posix() for p in sources}
    entries=[]
    for p in sorted(sources):
        name=p.relative_to(ROOT).as_posix();current=mapping[p.resolve()];text=p.read_text(encoding='utf-8');content=render(text)
        def rewrite(m):
            attr,target=m[1],html.unescape(m[2]);url=urlsplit(target)
            if url.scheme or url.netloc or not url.path:return m[0]
            original=(p.parent/unquote(url.path)).resolve()
            if original.is_dir():original=original/'README.md'
            if original in mapping:dest=relative(mapping[original],current)
            elif original.is_relative_to(ANALYSIS/'figures'):dest=relative('figures/'+original.name,current)
            elif original==ROOT/'ShopSimulator/assets/logo.png':dest=relative('assets/shopsimulator-logo.png',current)
            elif original.is_relative_to(ROOT):return f'{attr}="{html.escape(REPO+original.relative_to(ROOT).as_posix())}{("#"+url.fragment) if url.fragment else ""}"'
            else:raise ValueError('Reference escapes repository: '+target)
            return f'{attr}="{html.escape(dest)}{("#"+url.fragment) if url.fragment else ""}"'
        content=re.sub(r'(href|src)="([^"]+)"',rewrite,content)
        title=re.search(r'^#\s+(.+)',text,re.M);title=title[1] if title else name
        copy(p,'sources/'+name)
        emit(current,shell(current,title,'<article data-language="original">'+content+'</article>',source=name))
        entries.append(f'<div class="library-entry"><a href="{current}">{html.escape(title)}</a><small>{name}</small></div>')
    body='<article data-language="original"><h1 data-en="Reference library" data-zh="项目资料库">Reference library</h1><p data-en="Full source documents, rendered locally. Search by title or path. Archive entries describe historical plans and context." data-zh="完整源文档在站内渲染，可按标题或路径搜索。Archive 条目记录历史计划与背景。">Full source documents, rendered locally. Search by title or path. Archive entries describe historical plans and context.</p><label for="library-search" data-en="Search documents" data-zh="搜索文档">Search documents</label><input id="library-search" class="library-search" type="search">'+''.join(entries)+'</article>'
    emit('library.html',shell('library.html','Reference library',body))
    emit('.nojekyll','')
    manifest={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sorted(FILES.items())}
    (OUTPUT/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    unexpected=[p for p in OUTPUT.rglob('*') if p.is_file() and p.relative_to(OUTPUT).as_posix() not in FILES and p.name!='build-manifest.json']
    if unexpected:raise ValueError('Unexpected build files: '+str(unexpected))
    print(f'Built {len(FILES)} files, 6 bilingual chapters, {len(sources)} source documents.')

if __name__=='__main__':main()
