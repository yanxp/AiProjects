/**
 * 互联网嘴替测试 — 测试引擎
 */

/* ── State ── */
let currentQuestion = 0;
let answers = [];
let dimensionScores = {};

// DIMENSION_KEYS is defined in types.js

/* ── Page Navigation ── */
function showPage(id) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function goHome() {
  showPage('landing');
}

/* ── Landing: Render Type Cards ── */
function renderTypeCards() {
  const grid = document.getElementById('typesGrid');
  if (!grid) return;
  grid.innerHTML = SBTI_TYPES.map(t => `
    <div class="type-card-mini" onclick="showTypeModal('${t.code}')">
      <div class="tc-emoji">${t.emoji}</div>
      <div class="tc-code">${t.code}</div>
      <div class="tc-name">${t.name}</div>
    </div>
  `).join('');
}

function showTypeModal(code) {
  const t = SBTI_TYPES.find(x => x.code === code);
  if (!t) return;

  const existing = document.querySelector('.type-modal-overlay');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.className = 'type-modal-overlay';
  overlay.onclick = e => { if (e.target === overlay) overlay.remove(); };
  overlay.innerHTML = `
    <div class="type-modal">
      <button class="modal-close" onclick="this.closest('.type-modal-overlay').remove()">&times;</button>
      <div class="tm-emoji">${t.emoji}</div>
      <div class="tm-code">${t.code}</div>
      <div class="tm-name">${t.name} · ${t.nameEn}</div>
      <div class="tm-tagline">"${t.tagline}"</div>
      <div class="tm-desc">${t.desc}</div>
    </div>
  `;
  document.body.appendChild(overlay);
}

/* ── Quiz Flow ── */
function startQuiz() {
  currentQuestion = 0;
  answers = [];
  dimensionScores = {};
  DIMENSION_KEYS.forEach(k => { dimensionScores[k] = 0; });
  showPage('quiz');
  renderQuestion();
}

function renderQuestion() {
  const q = QUESTIONS[currentQuestion];
  if (!q) return;

  document.getElementById('progressText').textContent = `${currentQuestion + 1} / ${QUESTIONS.length}`;
  document.getElementById('progressFill').style.width = `${((currentQuestion + 1) / QUESTIONS.length) * 100}%`;
  document.getElementById('questionModel').textContent = q.model;
  document.getElementById('questionText').textContent = q.text;

  const letters = ['A', 'B', 'C', 'D'];
  const list = document.getElementById('optionsList');
  list.innerHTML = q.options.map((opt, i) => `
    <button class="option-btn" onclick="selectOption(${i}, ${opt.score})">
      <span class="opt-letter">${letters[i]}</span>
      <span>${opt.text}</span>
    </button>
  `).join('');
}

function selectOption(index, score) {
  const q = QUESTIONS[currentQuestion];

  const btns = document.querySelectorAll('.option-btn');
  btns.forEach((b, i) => {
    b.classList.toggle('selected', i === index);
    b.style.pointerEvents = 'none';
  });

  answers.push({ dimension: q.dimension, score });
  dimensionScores[q.dimension] = (dimensionScores[q.dimension] || 0) + score;

  setTimeout(() => {
    currentQuestion++;
    if (currentQuestion < QUESTIONS.length) {
      transitionQuestion();
    } else {
      showLoading();
    }
  }, 350);
}

function transitionQuestion() {
  const area = document.getElementById('questionArea');
  area.classList.add('slide-out');
  setTimeout(() => {
    area.classList.remove('slide-out');
    renderQuestion();
    area.classList.add('slide-in');
    setTimeout(() => area.classList.remove('slide-in'), 300);
  }, 280);
}

/* ── Loading Animation ── */
function showLoading() {
  showPage('loading');
  const emojis = ['🗡️', '🍉', '⌨️', '🎭', '💢', '🤣', '📢', '🔍'];
  const sub = document.getElementById('loadingSub');
  const emojiEl = document.getElementById('loadingEmoji');
  let step = 0;
  const messages = [
    '正在扫描你的互联网DNA...',
    '分析你的表达风格...',
    '检测阴阳浓度...',
    '评估对线战斗力...',
    '测量吃瓜指数...',
    '解析网络人设...',
    '匹配网络人格类型...',
    '即将揭晓你的嘴替身份！'
  ];

  const interval = setInterval(() => {
    if (step < messages.length) {
      sub.textContent = messages[step];
      emojiEl.textContent = emojis[step % emojis.length];
      step++;
    }
  }, 400);

  setTimeout(() => {
    clearInterval(interval);
    calculateResult();
  }, 3200);
}

/* ── Scoring ── */
function calculateResult() {
  const levels = {};
  DIMENSION_KEYS.forEach(k => {
    const s = dimensionScores[k] || 4;
    if (s <= 3) levels[k] = 'L';
    else if (s <= 5) levels[k] = 'M';
    else levels[k] = 'H';
  });

  let bestType = SBTI_TYPES[0];
  let bestScore = -1;

  SBTI_TYPES.forEach(t => {
    let matchScore = 0;
    let totalDims = 0;
    for (const dim in t.pattern) {
      totalDims++;
      const expected = t.pattern[dim];
      const actual = levels[dim];
      if (actual === expected) {
        matchScore += 3;
      } else if (
        (actual === 'M' && (expected === 'L' || expected === 'H')) ||
        (expected === 'M' && (actual === 'L' || actual === 'H'))
      ) {
        matchScore += 1;
      }
    }
    const normalized = totalDims > 0 ? matchScore / totalDims : 0;
    const finalScore = normalized + (matchScore * 0.05);
    if (finalScore > bestScore) {
      bestScore = finalScore;
      bestType = t;
    }
  });

  showResult(bestType, levels);
}

/* ── Result Display ── */
function showResult(type, levels) {
  showPage('result');

  document.getElementById('resultEmoji').textContent = type.emoji;
  document.getElementById('resultCode').textContent = type.code;
  document.getElementById('resultName').textContent = `${type.name} · ${type.nameEn}`;
  document.getElementById('resultTagline').textContent = `"${type.tagline}"`;
  document.getElementById('resultDesc').textContent = type.desc;

  const traitsEl = document.getElementById('resultTraits');
  traitsEl.innerHTML = type.traits.map(t => `<span class="trait-tag">${t}</span>`).join('');

  const dimGrid = document.getElementById('dimGrid');
  dimGrid.innerHTML = DIMENSION_KEYS.map(k => {
    const lv = levels[k];
    let pct, label, cls;
    if (lv === 'L') { pct = 25; label = '低'; cls = 'low'; }
    else if (lv === 'M') { pct = 55; label = '中'; cls = 'mid'; }
    else { pct = 85; label = '高'; cls = 'high'; }
    return `
      <div class="dim-item">
        <span class="dim-label">${DIMENSION_LABELS[k]}</span>
        <div class="dim-bar"><div class="dim-fill" style="width:${pct}%"></div></div>
        <span class="dim-level ${cls}">${label}</span>
      </div>`;
  }).join('');

  window._currentResult = { type, levels };
}

/* ── Share ── */
function shareResult() {
  const r = window._currentResult;
  if (!r) return;
  const text = `【互联网嘴替测试】我的网络人格是 ${r.type.emoji} ${r.type.code}（${r.type.name}）！\n"${r.type.tagline}"\n快来测测你是哪种网络人格 →`;

  if (navigator.share) {
    navigator.share({ title: '互联网嘴替测试', text }).catch(() => {});
  } else {
    fallbackCopy(text);
  }
}

function fallbackCopy(text) {
  function showCopied() {
    const btn = document.querySelector('.btn-share');
    if (!btn) return;
    const origHTML = btn.innerHTML;
    btn.textContent = '已复制到剪贴板！';
    setTimeout(() => { btn.innerHTML = origHTML; }, 2000);
  }
  function execCopy() {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    showCopied();
  }
  if (navigator.clipboard) {
    navigator.clipboard.writeText(text).then(showCopied).catch(execCopy);
  } else {
    execCopy();
  }
}

/* ── Init ── */
document.addEventListener('DOMContentLoaded', () => {
  renderTypeCards();
});
