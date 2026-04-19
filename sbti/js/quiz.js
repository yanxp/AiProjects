/**
 * SBTI Quiz Engine
 * Handles quiz flow, scoring, type matching, and result display.
 */

/* ── State ── */
let currentQuestion = 0;
let answers = [];          // [{dimension, score}, ...]
let dimensionScores = {};  // { selfEsteem: 6, ... }

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

  // Visual feedback
  const btns = document.querySelectorAll('.option-btn');
  btns.forEach((b, i) => {
    b.classList.toggle('selected', i === index);
    b.style.pointerEvents = 'none';
  });

  // Record answer
  answers.push({ dimension: q.dimension, score });
  dimensionScores[q.dimension] = (dimensionScores[q.dimension] || 0) + score;

  // Next question or finish
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
  const emojis = ['🔮', '🧠', '💀', '🐒', '🎮', '💩', '🤡', '👻'];
  const sub = document.getElementById('loadingSub');
  const emojiEl = document.getElementById('loadingEmoji');
  let step = 0;
  const messages = [
    '正在扫描你的灵魂...',
    '分析自我模型中...',
    '解析情感维度...',
    '计算态度倾向...',
    '测量行动力指数...',
    '评估社交风格...',
    '匹配人格类型...',
    '即将揭晓结果！'
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

/* ── Scoring & Matching ── */
function calculateResult() {
  // Normalize dimension scores to L/M/H
  // Each dimension has 2 questions, score range: 2-8
  const levels = {};
  DIMENSION_KEYS.forEach(k => {
    const s = dimensionScores[k] || 4; // default mid
    if (s <= 3) levels[k] = 'L';
    else if (s <= 5) levels[k] = 'M';
    else levels[k] = 'H';
  });

  // Match against type patterns — find best match
  let bestType = SBTI_TYPES[0];
  let bestScore = -1;

  SBTI_TYPES.forEach(t => {
    let matchScore = 0;
    let totalDims = 0;
    for (const [dim, expected] of Object.entries(t.pattern)) {
      totalDims++;
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
    // Normalize by total dimensions to not favor types with fewer pattern keys
    const normalized = totalDims > 0 ? matchScore / totalDims : 0;
    // Add small bonus for types with more matching dimensions (richer patterns)
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

  // Traits
  const traitsEl = document.getElementById('resultTraits');
  traitsEl.innerHTML = type.traits.map(t => `<span class="trait-tag">${t}</span>`).join('');

  // Dimension bars
  const grid = document.getElementById('dimensionGrid');
  grid.innerHTML = DIMENSION_KEYS.map(k => {
    const raw = dimensionScores[k] || 4;
    const pct = Math.round(((raw - 2) / 6) * 100);
    const level = levels[k];
    const levelLabel = level === 'H' ? '高' : level === 'M' ? '中' : '低';
    return `
      <div class="dim-item">
        <span class="dim-label">${DIMENSION_LABELS[k]}</span>
        <div class="dim-bar-wrap">
          <div class="dim-bar-fill" style="width: 0%;" data-width="${pct}%"></div>
        </div>
        <span class="dim-score">${levelLabel}</span>
      </div>
    `;
  }).join('');

  // Animate bars
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.querySelectorAll('.dim-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.width;
      });
    }, 100);
  });
}

/* ── Actions ── */
function retakeQuiz() {
  startQuiz();
}

function shareResult() {
  const code = document.getElementById('resultCode').textContent;
  const name = document.getElementById('resultName').textContent;
  const tagline = document.getElementById('resultTagline').textContent;

  const shareText = `我的SBTI人格类型是 ${code}（${name}）！${tagline}\n来测测你的沙雕人格 👉`;

  if (navigator.share) {
    navigator.share({
      title: 'SBTI 沙雕大测试',
      text: shareText,
      url: window.location.href
    }).catch(() => {
      copyToClipboard(shareText);
    });
  } else {
    copyToClipboard(shareText);
  }
}

function copyToClipboard(text) {
  if (navigator.clipboard) {
    navigator.clipboard.writeText(text).then(() => {
      showToast('已复制到剪贴板，快去分享给朋友吧！');
    }).catch(() => {
      fallbackCopy(text);
    });
  } else {
    fallbackCopy(text);
  }
}

function fallbackCopy(text) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.cssText = 'position:fixed;opacity:0';
  document.body.appendChild(ta);
  ta.select();
  document.execCommand('copy');
  document.body.removeChild(ta);
  showToast('已复制到剪贴板！');
}

function showToast(msg) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2500);
}

/* ── Init ── */
document.addEventListener('DOMContentLoaded', () => {
  renderTypeCards();
});
