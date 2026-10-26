(() => {
  'use strict';
  const root = document.documentElement;
  function saved(key) { try { return localStorage.getItem(key); } catch { return null; } }
  function save(key, value) { try { localStorage.setItem(key, value); } catch {} }
  const theme = document.getElementById('theme-toggle');
  const size = document.getElementById('text-toggle');
  const solutions = document.getElementById('solutions-toggle');
  function setTheme(light) {
    root.classList.toggle('light', light);
    if (theme) { theme.textContent = light ? 'Dark theme' : 'Light theme'; theme.setAttribute('aria-pressed', String(light)); }
  }
  function setSize(large) {
    root.classList.toggle('large-text', large);
    if (size) { size.textContent = large ? 'Standard text' : 'Larger text'; size.setAttribute('aria-pressed', String(large)); }
  }
  setTheme(saved('6s184-theme') === 'light'); setSize(saved('6s184-size') === 'large');
  theme?.addEventListener('click', () => { const light = !root.classList.contains('light'); setTheme(light); save('6s184-theme', light ? 'light' : 'dark'); });
  size?.addEventListener('click', () => { const large = !root.classList.contains('large-text'); setSize(large); save('6s184-size', large ? 'large' : 'normal'); });
  solutions?.addEventListener('click', () => {
    const open = solutions.getAttribute('aria-pressed') !== 'true';
    document.querySelectorAll('details.solution').forEach(e => { e.open = open; });
    solutions.setAttribute('aria-pressed', String(open)); solutions.textContent = open ? 'Collapse solutions' : 'Expand solutions';
  });
  let restorePrint = [];
  addEventListener('beforeprint', () => { restorePrint = [...document.querySelectorAll('details')].map(e => [e, e.open]); restorePrint.forEach(([e]) => { e.open = true; }); });
  addEventListener('afterprint', () => restorePrint.forEach(([e, open]) => { e.open = open; }));
  document.getElementById('print-page')?.addEventListener('click', () => print());
  const toc = document.getElementById('chapter-toc');
  if (toc) document.querySelectorAll('#chapter-content h2[id]').forEach(h => {
    const a = document.createElement('a'); a.href = '#' + h.id;
    a.textContent = h.childNodes[0]?.textContent?.trim() || h.textContent; toc.append(a);
  });
  const input = document.getElementById('lecture-search');
  input?.addEventListener('input', () => {
    const words = input.value.toLowerCase().trim().split(/\s+/); let count = 0;
    document.querySelectorAll('[data-chapter]').forEach(e => { e.hidden = !words.every(w => e.textContent.toLowerCase().includes(w)); if (!e.hidden) count++; });
    document.getElementById('search-status').textContent = count + ' chapters shown';
  });
  document.querySelectorAll('pre').forEach(pre => {
    let wrap = pre.parentElement;
    if (!wrap.classList.contains('codewrap')) { wrap = document.createElement('div'); wrap.className = 'codewrap'; pre.before(wrap); wrap.append(pre); }
    let b = wrap.querySelector('.copybtn');
    if (!b) { b = document.createElement('button'); b.type = 'button'; b.className = 'copybtn'; b.textContent = 'Copy'; wrap.append(b); }
    b.addEventListener('click', async () => {
      const text = pre.querySelector('code')?.textContent || pre.textContent;
      try {
        if (navigator.clipboard && isSecureContext) await navigator.clipboard.writeText(text);
        else { const t = document.createElement('textarea'); t.value = text; t.style.position = 'fixed'; t.style.opacity = '0'; document.body.append(t); t.select(); const ok = document.execCommand('copy'); t.remove(); if (!ok) throw Error('Copy unavailable'); }
        b.textContent = 'Copied';
      } catch { b.textContent = 'Select text to copy'; }
      setTimeout(() => { b.textContent = 'Copy'; }, 1500);
    });
  });
})();
