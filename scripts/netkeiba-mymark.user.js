// ==UserScript==
// @name         KeibaAI マイ印
// @namespace    keibaai
// @version      0.1
// @description  KeibaAI の「netkeiba で印を付ける」ボタンで開いた出馬表に、予想タブの印を入れる
// @match        https://race.netkeiba.com/race/shutuba.html*
// @grant        none
// @run-at       document-idle
// ==/UserScript==

// 印はアドレスの #keibaai=馬番-印番号,... から読む(1=◎ 2=○ 3=▲ 4=△ 5=☆)。
// netkeiba の画面の印プルダウン(li)を、手で選ぶのと同じように1頭ずつクリックする。通信は netkeiba の画面がふだん通り行う。
// 印を付けない馬に印が残っていれば「--」に戻す。読み終えたら # 以降を消し、再読み込みで入れ直さない。
(function () {
  'use strict';
  const m = location.hash.match(/keibaai=([0-9,\-]*)/);
  if (!m) return;
  const want = {};
  for (const p of m[1].split(',')) {
    const [n, k] = p.split('-');
    if (n && k) want[n] = k;
  }
  history.replaceState(null, '', location.pathname + location.search);

  const NAMES = { 1: '◎', 2: '○', 3: '▲', 4: '△', 5: '☆' };
  const banner = (text, ok) => {
    let b = document.getElementById('keibaai-banner');
    if (!b) {
      b = document.createElement('div');
      b.id = 'keibaai-banner';
      b.style.cssText = 'position:fixed;top:8px;right:8px;z-index:99999;padding:8px 12px;border-radius:6px;'
        + 'font:14px sans-serif;color:#fff;box-shadow:0 2px 6px rgba(0,0,0,.3)';
      document.body.appendChild(b);
    }
    b.style.background = ok ? '#2e7d32' : '#c62828';
    b.textContent = text;
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // 馬番 → { select, items(li の並び) }。プルダウンは netkeiba の画面が後から作るので待つ。
  const rows = () => {
    const out = {};
    for (const sel of document.querySelectorAll('select[id^="mark_"]')) {
      const opt = sel.querySelector('option');
      const box = sel.parentElement.querySelector('.tzSelect');
      if (!opt || !box) continue;
      out[opt.value.split('_')[0]] = { sel, items: box.querySelectorAll('ul.dropDown li') };
    }
    return out;
  };

  (async () => {
    banner('KeibaAI: 印を入れる準備中…', true);
    let r = {};
    for (let i = 0; i < 40; i++) {
      r = rows();
      if (Object.keys(r).length && document.querySelectorAll('select[id^="mark_"]').length === Object.keys(r).length) break;
      await sleep(250);
    }
    if (!Object.keys(r).length) {
      banner('KeibaAI: 印のプルダウンが見つかりません(印モードに「切替」してから、もう一度ボタンを押してください)', false);
      return;
    }
    await sleep(1500); // 保存済みの印を画面が読み込むのを待つ
    const missing = Object.keys(want).filter((n) => !r[n]);
    let changed = 0;
    for (const [n, row] of Object.entries(r)) {
      const target = want[n] || '0';
      const current = (row.sel.value || '').split('_')[1] || '0';
      if (current === target) continue;
      if (!want[n] && (current === '98' || current === '99')) continue; // ✓・消 は尚さんの印として残す
      const idx = [...row.sel.options].findIndex((o) => o.value.split('_')[1] === target);
      if (idx < 0 || !row.items[idx]) continue;
      row.items[idx].click();
      changed++;
      await sleep(400);
    }
    const summary = Object.entries(want).map(([n, k]) => `${NAMES[k] || '?'}${n}`).join(' ');
    if (missing.length) {
      banner(`KeibaAI: ${summary}(馬番 ${missing.join(',')} が出馬表にありません)`, false);
    } else {
      banner(`KeibaAI: 印を入れました ${summary}(変更 ${changed} 頭)`, true);
      setTimeout(() => document.getElementById('keibaai-banner')?.remove(), 8000);
    }
  })();
})();
