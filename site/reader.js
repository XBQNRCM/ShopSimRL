'use strict';
let readerLanguage='en';
try { readerLanguage=localStorage.getItem('intraSkillLanguage')==='zh'?'zh':'en'; } catch (_) {}
function readerTranslate(lang) {
  readerLanguage=lang;
  document.documentElement.lang=lang==='zh'?'zh-CN':'en';
  document.querySelectorAll('[data-en]').forEach(el=>{el.textContent=el.dataset[lang];});
  document.querySelectorAll('[data-language]').forEach(el=>{el.hidden=el.dataset.language!=='original'&&el.dataset.language!==lang;});
  const button=document.getElementById('language');button.textContent=lang==='zh'?'English':'中文';
  button.setAttribute('aria-label',lang==='zh'?'Switch to English':'切换至中文');
  try { localStorage.setItem('intraSkillLanguage',lang); } catch (_) {}
  const outline=document.getElementById('outline');
  if(outline)outline.replaceChildren(...Array.from(document.querySelectorAll('article:not([hidden]) h2')).map(h=>{
    const a=document.createElement('a');a.href='#'+h.id;a.textContent=h.textContent;return a;
  }));
  if(window.MathJax?.typesetPromise)window.MathJax.typesetPromise().catch(()=>{});
}
document.getElementById('language').addEventListener('click',()=>readerTranslate(readerLanguage==='en'?'zh':'en'));
readerTranslate(readerLanguage);
function followAnchor() {
  let id;try{id=decodeURIComponent(location.hash.slice(1));}catch(_){return;}
  const node=document.getElementById(id),article=node?.closest('[data-language]');
  if(article?.hidden)readerTranslate(article.dataset.language);
  if(node)node.scrollIntoView();
}
if(location.hash)followAnchor();window.addEventListener('hashchange',followAnchor);
const search=document.getElementById('library-search');
if(search)search.addEventListener('input',()=>{
  const query=search.value.trim().toLocaleLowerCase();
  document.querySelectorAll('.library-entry').forEach(el=>{el.hidden=!el.textContent.toLocaleLowerCase().includes(query);});
});
