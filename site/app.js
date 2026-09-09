'use strict';
let language = 'en';
let analysis = null;
let behavior = null;
const $ = id => document.getElementById(id);
const tr = (en, zh) => language === 'zh' ? zh : en;
const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function setLanguage(value) {
  language = value;
  document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
  document.querySelectorAll('[data-en]').forEach(node => {
    const text = node.dataset[language];
    node.replaceChildren(...text.split('\\n').flatMap((line, i) => i ? [document.createElement('br'), document.createTextNode(line)] : [document.createTextNode(line)]));
  });
  $('language').textContent = tr('中文', 'English');
  $('language').setAttribute('aria-label', tr('切换至中文', 'Switch to English'));
  try { localStorage.setItem('intraSkillLanguage', language); } catch (_) { /* preference storage is optional */ }
  renderChart(); renderSkills(); renderBehavior();
}

function renderChart() {
  if (!analysis) return;
  const metric = $('metric').value;
  const values = analysis.training_steps.map(row => row['rollout/shopsim/' + metric]);
  const smooth = values.map((_,i) => { const a=values.slice(Math.max(0,i-4),i+1); return a.reduce((s,v)=>s+v,0)/a.length; });
  const width=1000, height=325, left=55, right=25, top=25, bottom=45;
  const xx=i=>left+i/79*(width-left-right), yy=v=>top+(1-v)*(height-top-bottom);
  const path=v=>v.map((y,i)=>`${i?'L':'M'}${xx(i).toFixed(2)},${yy(y).toFixed(2)}`).join(' ');
  const active=Number($('step').value)-1;
  const value=values[active];
  const grid=[0,.25,.5,.75,1].map(v=>`<line x1="${left}" x2="${width-right}" y1="${yy(v)}" y2="${yy(v)}" stroke="#e5ebf4"/><text x="${left-12}" y="${yy(v)+4}" text-anchor="end" font-size="13" fill="#52627a">${v.toFixed(2)}</text>`).join('');
  const ticks=[1,20,40,60,80].map(n=>`<text x="${xx(n-1)}" y="${height-20}" text-anchor="middle" font-size="13" fill="#52627a">${n}</text>`).join('');
  const bounds=[19.5,39.5,59.5].map(n=>`<line x1="${xx(n)}" x2="${xx(n)}" y1="${top}" y2="${height-bottom}" stroke="#cdd8ea" stroke-dasharray="5 5"/>`).join('');
  const q=metric==='skill_free_group_rate'?`<line x1="${left}" x2="${width-right}" y1="${yy(.2)}" y2="${yy(.2)}" stroke="#8246c6" stroke-dasharray="4 4"/>`:'';
  $('training-chart').innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="chart-title"><title id="chart-title">${escapeHTML($('metric').selectedOptions[0].textContent)}: ${tr('steps 1 to 80','第 1 至 80 步')}</title>${grid}${ticks}${bounds}${q}<path d="${path(values)}" fill="none" stroke="#245ce5" opacity=".23" stroke-width="2"/><path d="${path(smooth)}" fill="none" stroke="#245ce5" stroke-width="3"/><line x1="${xx(active)}" x2="${xx(active)}" y1="${top}" y2="${height-bottom}" stroke="#8246c6" stroke-width="1"/><circle cx="${xx(active)}" cy="${yy(value)}" r="5" fill="#8246c6"/><text x="${width/2}" y="${height-1}" text-anchor="middle" fill="#52627a" font-size="12">${tr('Rollout step','Rollout 步数')}</text></svg>`;
  $('step-output').textContent=tr(`Step ${active+1} · ${value.toFixed(4)}`,`第 ${active+1} 步 · ${value.toFixed(4)}`);
}

function renderSkills() {
  if (!analysis) return;
  const index=Number($('bank').value), bank=analysis.skill_banks[index];
  for (let i=0;i<5;i++) $('bank').options[i].textContent=`${i===4?'Sₜ':'S'+i} · ${tr('Step','步数')} ${analysis.skill_banks[i].step} · ${analysis.skill_banks[i].skills.length} ${tr('chunks','技能块')}`;
  $('skill-list').replaceChildren(...bank.skills.map(skill=>{
    const details=document.createElement('details');details.className='skill';
    const summary=document.createElement('summary');
    const title=document.createElement('span');title.textContent=skill.content.split('\n')[0].replace(/^#+\s*/,'');
    const id=document.createElement('span');id.className='skill-id';id.textContent=skill.skill_id;title.append(id);
    const coefficient=document.createElement('span');coefficient.className='coefficient';
    coefficient.textContent=`β = ${Number(skill.metadata.estimated_effect).toFixed(4)}`;
    summary.append(title,coefficient);details.append(summary);
    const text=document.createElement('pre');text.lang='zh-CN';text.textContent=skill.content;details.append(text);return details;
  }));
}

$('language').addEventListener('click',()=>setLanguage(language==='en'?'zh':'en'));
$('metric').addEventListener('change',renderChart);
$('step').addEventListener('input',renderChart);
$('bank').addEventListener('change',renderSkills);
for(const id of ['behavior-cohort','behavior-outcome','behavior-metric','behavior-stat'])$(id).addEventListener('change',renderBehavior);
try { language=localStorage.getItem('intraSkillLanguage')==='zh'?'zh':'en'; } catch (_) {}
setLanguage(language);
fetch('data/analysis.json').then(response=>{if(!response.ok)throw new Error('Analysis unavailable');return response.json();}).then(data=>{
  analysis=data;renderChart();renderSkills();
}).catch(()=>{
  const message='Interactive data could not load. Download the dataset or view the static figures. / 交互数据加载失败，请下载数据或查看静态图表。';
  for(const id of ['training-chart','skill-list']) {$(id).className='error';$(id).textContent=message;}
});

function renderBehavior() {
  if(!behavior)return;
  const cohort=$('behavior-cohort').value,outcome=$('behavior-outcome').value,metric=$('behavior-metric').value,stat=$('behavior-stat').value;
  const rows=cohort==='train'?behavior.steps.map(r=>({step:r.step,data:r[outcome]})):
    ['val_base','val_20','val_40','val_60','val_80'].map((c,i)=>({step:i*20,data:outcome==='all'?behavior.evaluation[c]:behavior.evaluation_outcomes[c][outcome]}));
  const points=rows.map(r=>({step:r.step,n:r.data.n,value:r.data.metrics[metric][stat]})).filter(r=>r.value!==null);
  const width=1000,height=340,left=70,right=25,top=25,bottom=45,max=Math.max(.01,...points.map(r=>r.value))*1.12;
  const xx=s=>left+s/80*(width-left-right),yy=v=>top+(1-v/max)*(height-top-bottom);
  const grid=Array.from({length:5},(_,i)=>{const v=max*i/4;return `<line x1="${left}" x2="${width-right}" y1="${yy(v)}" y2="${yy(v)}" stroke="#e5ebf4"/><text x="${left-10}" y="${yy(v)+4}" text-anchor="end" fill="#52627a" font-size="13">${v.toFixed(max>20?0:2)}</text>`;}).join('');
  const ticks=[0,20,40,60,80].map(s=>`<text x="${xx(s)}" y="${height-20}" text-anchor="middle" fill="#52627a" font-size="13">${s}</text>`).join('');
  const path=points.map((r,i)=>`${i?'L':'M'}${xx(r.step)},${yy(r.value)}`).join(' ');
  const title=`${$('behavior-cohort').selectedOptions[0].textContent} · ${$('behavior-outcome').selectedOptions[0].textContent} · ${$('behavior-metric').selectedOptions[0].textContent} · ${$('behavior-stat').selectedOptions[0].textContent}`;
  const circles=points.map(r=>`<circle cx="${xx(r.step)}" cy="${yy(r.value)}" r="${cohort==='val'?5:2}" fill="#8246c6"><title>${r.step}: ${r.value.toFixed(3)} (n=${r.n})</title></circle>`).join('');
  $('behavior-chart').innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="behavior-title"><title id="behavior-title">${escapeHTML(title)}</title>${grid}${ticks}<path d="${path}" stroke="#8246c6" stroke-width="2" fill="none"/>${circles}<text x="${width/2}" y="${height-1}" text-anchor="middle" fill="#52627a" font-size="12">${tr('Rollout step / checkpoint','Rollout 步数 / checkpoint')}</text></svg>`;
  const first=points[0],last=points.at(-1);
  $('behavior-summary').textContent=`${title}: ${first.value.toFixed(3)} → ${last.value.toFixed(3)}. `+tr('Raw values; no smoothing. Endpoint change is descriptive, not a paired effect estimate.','原始数值，未平滑。端点变化仅作描述，不是配对效应估计。');
  $('behavior-values').innerHTML=`<table><caption>${escapeHTML(title)}</caption><thead><tr><th>${tr('Step','步数')}</th><th>n</th><th>${tr('Value','数值')}</th></tr></thead><tbody>${points.map(r=>`<tr><td>${r.step}</td><td>${r.n}</td><td>${r.value.toFixed(4)}</td></tr>`).join('')}</tbody></table>`;
}
fetch('data/behavior/behavior.json').then(r=>{if(!r.ok)throw new Error('Behavior unavailable');return r.json();}).then(d=>{behavior=d;renderBehavior();}).catch(()=>{
  $('behavior-chart').textContent='Interactive data unavailable; see the full report and static figures. / 交互数据不可用，请查看完整报告与静态图。';
});
