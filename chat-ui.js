/*
 * chat-ui.js -- render Markdown + tabel rapi + efek mengetik untuk chatbot dashboard.
 *
 * Butuh dua library (taruh SEBELUM file ini di index.html):
 *   <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.3.0/marked.min.js"></script>
 *   <script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.0.6/purify.min.js"></script>
 * Tanpa keduanya modul tetap jalan, tapi jawaban tampil sebagai teks polos.
 *
 * API (global ChatUI):
 *   ChatUI.showTyping(container)            -> tiga titik animasi saat menunggu server; return fungsi untuk menghapusnya
 *   ChatUI.typeInto(el, markdown, opts)     -> mengetik jawaban ke elemen el; return Promise (selesai saat teks lengkap)
 *   ChatUI.renderInto(el, markdown, opts)   -> render langsung tanpa animasi (untuk riwayat chat)
 *   opts: { scrollEl, speed (ms/tick, default 18), maxMs (durasi maks, default 7000), instant, skipOnClick }
 */
(function (global) {
  'use strict';

  var NARROW_PX = 560;
  var reduceMotion = !!(global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches);

  function escapeHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // Tabel: bungkus agar bisa digulir, beri label per sel (dipakai tampilan kartu di layar sempit)
  function decorateTable(table) {
    var heads = Array.prototype.map.call(table.querySelectorAll('thead th'), function (th) {
      return th.textContent.trim();
    });
    var noIdx = -1, nameIdx = -1;
    heads.forEach(function (h, i) {
      if (noIdx < 0 && /^(no\.?|#)$/i.test(h)) noIdx = i;
      if (nameIdx < 0 && /(rumah sakit|nama)/i.test(h)) nameIdx = i;
    });
    Array.prototype.forEach.call(table.querySelectorAll('tr'), function (tr) {
      Array.prototype.forEach.call(tr.children, function (cell, i) {
        if (i === noIdx) cell.classList.add('col-no');
        if (i === nameIdx) cell.classList.add('col-name');
        if (cell.tagName === 'TD' && heads[i]) cell.setAttribute('data-label', heads[i]);
      });
    });
    var wrap = document.createElement('div');
    wrap.className = 'tbl-wrap';
    table.parentNode.replaceChild(wrap, table);
    wrap.appendChild(table);
  }

  function renderTemplate(markdown) {
    var md = String(markdown == null ? '' : markdown);
    var html;
    if (global.marked && global.DOMPurify) {
      var parse = global.marked.parse || global.marked;
      html = global.DOMPurify.sanitize(parse.call(global.marked, md, { gfm: true, breaks: true }));
    } else {
      html = '<p>' + escapeHtml(md).replace(/\n/g, '<br>') + '</p>';
    }
    var tpl = document.createElement('template');
    tpl.innerHTML = html;
    Array.prototype.forEach.call(tpl.content.querySelectorAll('a'), function (a) {
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
    });
    Array.prototype.forEach.call(tpl.content.querySelectorAll('table'), decorateTable);
    return tpl;
  }

  // Layar/jendela sempit -> tabel jadi kartu (lihat chat-ui.css)
  function applyLayout(el) {
    var host = el.__chatHost || el.parentElement || el;
    el.classList.toggle('is-narrow', host.clientWidth < NARROW_PX);
  }
  global.addEventListener('resize', function () {
    Array.prototype.forEach.call(document.querySelectorAll('.chat-md'), applyLayout);
  });

  function prepare(el, opts) {
    el.classList.add('chat-md');
    el.__chatHost = (opts && opts.scrollEl) || null;
  }

  function renderInto(el, markdown, opts) {
    prepare(el, opts);
    el.innerHTML = '';
    el.appendChild(renderTemplate(markdown).content);
    applyLayout(el);
    if (opts && opts.scrollEl) opts.scrollEl.scrollTop = opts.scrollEl.scrollHeight;
  }

  function typeInto(el, markdown, opts) {
    opts = opts || {};
    var speed = opts.speed || 18;
    var maxMs = opts.maxMs || 7000;
    var scrollEl = opts.scrollEl || null;
    function scroll() { if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight; }

    if (reduceMotion || opts.instant) {
      renderInto(el, markdown, opts);
      return Promise.resolve();
    }

    prepare(el, opts);
    var tpl = renderTemplate(markdown);
    var blocks = Array.prototype.slice.call(tpl.content.children);
    el.innerHTML = '';
    el.appendChild(tpl.content);
    applyLayout(el);

    // Susun antrean: tiap blok baru ditampilkan saat gilirannya; teks diketik per karakter,
    // baris tabel muncul satu per satu (lebar kolom stabil karena isi sudah ada, hanya disembunyikan).
    var queue = [];
    var totalChars = 0;
    blocks.forEach(function (block) {
      block.style.display = 'none';
      queue.push({ t: 'block', block: block });
      if (block.classList.contains('tbl-wrap')) {
        Array.prototype.forEach.call(block.querySelectorAll('tbody tr'), function (tr) {
          tr.style.visibility = 'hidden';
          queue.push({ t: 'row', tr: tr, wrap: block });
        });
        return;
      }
      Array.prototype.forEach.call(block.querySelectorAll('li'), function (li) { li.style.display = 'none'; });
      var walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
      var n;
      while ((n = walker.nextNode())) {
        if (!n.nodeValue.trim()) continue;
        var chars = Array.from(n.nodeValue);   // per code point, supaya emoji tidak terpotong
        queue.push({ t: 'text', node: n, chars: chars, full: n.nodeValue, block: block });
        n.nodeValue = '';
        totalChars += chars.length;
      }
    });

    var caret = document.createElement('span');
    caret.className = 'chat-caret';
    var chunk = Math.max(1, Math.ceil(totalChars / (maxMs / speed)));
    var qi = 0, pos = 0, timer = null, done = false, resolveFn = function () {};

    function showAncestors(node, block) {
      for (var p = node.parentNode; p && p !== block.parentNode; p = p.parentNode) {
        if (p.nodeType === 1) p.style.display = '';
      }
    }
    function finish() {
      if (done) return;
      done = true;
      clearTimeout(timer);
      if (caret.parentNode) caret.parentNode.removeChild(caret);
      resolveFn();
    }
    function revealAll() {
      if (done) return;
      queue.forEach(function (s) {
        if (s.t === 'block') s.block.style.display = '';
        else if (s.t === 'row') s.tr.style.visibility = '';
        else s.node.nodeValue = s.full;
      });
      Array.prototype.forEach.call(el.querySelectorAll('li'), function (li) { li.style.display = ''; });
      scroll();
      finish();
    }
    function tick() {
      if (done) return;
      var s = queue[qi];
      if (!s) { scroll(); finish(); return; }
      if (s.t === 'block') { s.block.style.display = ''; qi++; tick(); return; }
      if (s.t === 'row') {
        s.tr.style.visibility = '';
        qi++;
        s.wrap.parentNode.insertBefore(caret, s.wrap.nextSibling);
        scroll();
        timer = setTimeout(tick, 140);
        return;
      }
      if (pos === 0) showAncestors(s.node, s.block);
      pos = Math.min(s.chars.length, pos + chunk);
      s.node.nodeValue = s.chars.slice(0, pos).join('');
      if (s.node.parentNode) s.node.parentNode.insertBefore(caret, s.node.nextSibling);
      if (pos >= s.chars.length) { qi++; pos = 0; }
      scroll();
      timer = setTimeout(tick, speed);
    }

    if (opts.skipOnClick !== false) el.addEventListener('click', revealAll, { once: true });
    return new Promise(function (resolve) { resolveFn = resolve; tick(); });
  }

  function showTyping(container) {
    var d = document.createElement('div');
    d.className = 'chat-typing';
    d.setAttribute('aria-label', 'Asisten sedang menulis jawaban');
    d.innerHTML = '<span></span><span></span><span></span>';
    container.appendChild(d);
    container.scrollTop = container.scrollHeight;
    return function () { if (d.parentNode) d.parentNode.removeChild(d); };
  }

  global.ChatUI = { showTyping: showTyping, typeInto: typeInto, renderInto: renderInto, renderTemplate: renderTemplate };
})(window);
