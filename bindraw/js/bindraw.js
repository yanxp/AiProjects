/**
 * bindraw.js — 画板 + 手势驱动逻辑
 *
 * 画布直接覆盖在摄像头画面上（透明背景），指到哪里画到哪里。
 *
 * - 鼠标 / 触屏：画笔、橡皮擦、颜色、粗细、撤销、重做、清空、保存 PNG
 * - 手势驱动（订阅 window.BindrawCamera 的手势事件）：
 *     ☝️ point  → 下笔，以指尖位置连续作画
 *     🖐 none   → 抬笔（只显示光标）
 *   其它操作（撤销、保存、清空）通过工具栏完成，避免手势误触。
 * - 素描模板：可选择动物线稿作为底图，小朋友在上面着色。
 */
(function () {
    'use strict';

    const canvas = document.getElementById('draw-canvas');
    const ctx = canvas.getContext('2d');
    const statusEl = document.getElementById('draw-status');
    const cursorEl = document.getElementById('cursor-indicator');

    const btnPen = document.getElementById('tool-pen');
    const btnEraser = document.getElementById('tool-eraser');
    const btnUndo = document.getElementById('btn-undo');
    const btnRedo = document.getElementById('btn-redo');
    const btnClear = document.getElementById('btn-clear');
    const btnSave = document.getElementById('btn-save');
    const colorInput = document.getElementById('color-picker');
    const sizeSlider = document.getElementById('size-slider');
    const sizeValue = document.getElementById('size-value');
    const templatePicker = document.getElementById('template-picker');
    const gestureToggle = document.getElementById('gesture-toggle');
    const cameraToggle = document.getElementById('camera-toggle');
    const skeletonToggle = document.getElementById('skeleton-toggle');
    const legendToggle = document.getElementById('legend-toggle');
    const legendBody = document.getElementById('legend-body');

    // ---------- 动物素描模板（内联 SVG，viewBox 400x300，浅灰线稿） ----------
    const TEMPLATE_STROKE = '#9ca3af';
    const TEMPLATE_SVGS = {
        cat: `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
  <g fill="none" stroke="${TEMPLATE_STROKE}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <path d="M130 115 L115 50 L180 102 Z"/>
    <path d="M270 115 L285 50 L220 102 Z"/>
    <circle cx="200" cy="170" r="92"/>
    <circle cx="170" cy="162" r="9"/>
    <circle cx="230" cy="162" r="9"/>
    <path d="M192 192 Q200 205 208 192 Z"/>
    <path d="M200 205 Q200 215 200 218 M200 218 Q188 230 180 222 M200 218 Q212 230 220 222"/>
    <path d="M125 195 L170 200 M125 208 L170 205"/>
    <path d="M275 195 L230 200 M275 208 L230 205"/>
  </g>
</svg>`,
        rabbit: `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
  <g fill="none" stroke="${TEMPLATE_STROKE}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <ellipse cx="170" cy="65" rx="14" ry="48"/>
    <ellipse cx="230" cy="65" rx="14" ry="48"/>
    <circle cx="200" cy="150" r="52"/>
    <circle cx="181" cy="146" r="5"/>
    <circle cx="219" cy="146" r="5"/>
    <path d="M195 162 Q200 170 205 162 Z"/>
    <path d="M200 170 L200 180 M200 180 Q190 188 184 184 M200 180 Q210 188 216 184"/>
    <ellipse cx="200" cy="235" rx="58" ry="38"/>
    <ellipse cx="170" cy="272" rx="18" ry="9"/>
    <ellipse cx="230" cy="272" rx="18" ry="9"/>
  </g>
</svg>`,
        fish: `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
  <g fill="none" stroke="${TEMPLATE_STROKE}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <path d="M90 150 Q200 55 290 150 Q200 245 90 150 Z"/>
    <path d="M290 150 L355 95 L340 150 L355 205 Z"/>
    <circle cx="140" cy="135" r="9"/>
    <circle cx="140" cy="135" r="3" fill="${TEMPLATE_STROKE}"/>
    <path d="M175 108 Q183 150 175 192"/>
    <path d="M205 180 Q222 220 248 210"/>
    <path d="M205 90 Q235 68 252 88"/>
    <path d="M228 130 Q238 140 228 150"/>
    <path d="M250 122 Q260 135 250 148"/>
  </g>
</svg>`,
        butterfly: `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
  <g fill="none" stroke="${TEMPLATE_STROKE}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <ellipse cx="200" cy="150" rx="8" ry="60"/>
    <circle cx="200" cy="82" r="10"/>
    <path d="M195 74 Q178 56 172 44"/>
    <path d="M205 74 Q222 56 228 44"/>
    <path d="M192 110 Q90 80 95 165 Q140 160 192 140 Z"/>
    <path d="M208 110 Q310 80 305 165 Q260 160 208 140 Z"/>
    <path d="M192 155 Q105 180 130 245 Q170 230 192 190 Z"/>
    <path d="M208 155 Q295 180 270 245 Q230 230 208 190 Z"/>
    <circle cx="140" cy="122" r="8"/>
    <circle cx="260" cy="122" r="8"/>
    <circle cx="155" cy="210" r="6"/>
    <circle cx="245" cy="210" r="6"/>
  </g>
</svg>`,
        dinosaur: `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
  <g fill="none" stroke="${TEMPLATE_STROKE}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <path d="M80 220 Q80 150 150 140 Q250 130 310 150 Q340 160 340 200 Q340 235 310 240 L110 240 Q80 240 80 220 Z"/>
    <path d="M310 150 Q358 128 368 88 Q372 65 338 62 Q318 60 320 82 L322 105 Q318 125 308 142"/>
    <circle cx="350" cy="85" r="3" fill="${TEMPLATE_STROKE}"/>
    <path d="M345 100 L358 100"/>
    <path d="M130 240 L130 285 M180 240 L180 285 M240 240 L240 285 M290 240 L290 285"/>
    <path d="M130 145 L142 120 L154 146"/>
    <path d="M170 142 L184 112 L198 142"/>
    <path d="M210 140 L226 108 L240 140"/>
    <path d="M248 144 L262 112 L276 146"/>
    <path d="M80 208 Q42 216 20 240"/>
  </g>
</svg>`,
        turtle: `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
  <g fill="none" stroke="${TEMPLATE_STROKE}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <ellipse cx="200" cy="160" rx="110" ry="78"/>
    <path d="M200 82 L200 238"/>
    <path d="M128 130 L272 130 M120 170 L280 170 M136 208 L264 208"/>
    <path d="M158 100 L158 225 M242 100 L242 225"/>
    <ellipse cx="320" cy="155" rx="30" ry="22"/>
    <circle cx="332" cy="148" r="3" fill="${TEMPLATE_STROKE}"/>
    <path d="M308 158 L302 168"/>
    <ellipse cx="118" cy="220" rx="26" ry="12"/>
    <ellipse cx="282" cy="220" rx="26" ry="12"/>
    <ellipse cx="140" cy="105" rx="22" ry="10"/>
    <ellipse cx="260" cy="105" rx="22" ry="10"/>
    <path d="M88 160 L62 152 L68 170 Z"/>
  </g>
</svg>`
    };

    const MAX_HISTORY = 40;
    // 画画粘滞：进入 point 画画后，需要连续 N 帧非 point 才真正抬笔，
    // 避免识别偶发抖动导致断线。
    const DRAW_STICKY_FRAMES = 5;

    const state = {
        tool: 'pen',
        color: colorInput.value,
        size: parseInt(sizeSlider.value, 10),
        drawing: false,
        lastX: 0,
        lastY: 0,
        history: [],
        redoStack: [],
        gestureDrawing: false,
        smoothX: null,
        smoothY: null,
        activeGesture: 'none',
        nonPointFrames: 0,
        lastNormX: null,
        lastNormY: null,
        templateKey: 'none'
    };

    // ---------- Canvas 尺寸与历史 ----------

    function resizeCanvas(preserve) {
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const newW = Math.max(1, Math.floor(rect.width * dpr));
        const newH = Math.max(1, Math.floor(rect.height * dpr));
        if (canvas.width === newW && canvas.height === newH) return;

        let snapshot = null;
        if (preserve && canvas.width > 0 && canvas.height > 0) {
            snapshot = document.createElement('canvas');
            snapshot.width = canvas.width;
            snapshot.height = canvas.height;
            snapshot.getContext('2d').drawImage(canvas, 0, 0);
        }

        canvas.width = newW;
        canvas.height = newH;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        // 画布透明，不填充背景色（让摄像头画面透出来）
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (snapshot) {
            ctx.drawImage(snapshot, 0, 0, snapshot.width, snapshot.height, 0, 0, canvas.width, canvas.height);
        }
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
    }

    function saveHistory() {
        try {
            state.history.push(canvas.toDataURL('image/png'));
            if (state.history.length > MAX_HISTORY) state.history.shift();
            state.redoStack = [];
        } catch (err) {
            console.warn('[bindraw] saveHistory failed', err);
        }
    }

    function restoreFromDataURL(url) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                ctx.save();
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.globalCompositeOperation = 'source-over';
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                ctx.restore();
                resolve();
            };
            img.onerror = () => resolve();
            img.src = url;
        });
    }

    async function undo() {
        if (state.history.length <= 1) {
            setStatus('没有可撤销的操作');
            return;
        }
        const current = state.history.pop();
        state.redoStack.push(current);
        await restoreFromDataURL(state.history[state.history.length - 1]);
        setStatus('已撤销');
    }

    async function redo() {
        if (state.redoStack.length === 0) {
            setStatus('没有可重做的操作');
            return;
        }
        const url = state.redoStack.pop();
        state.history.push(url);
        await restoreFromDataURL(url);
        setStatus('已重做');
    }

    function clearCanvas() {
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.globalCompositeOperation = 'source-over';
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
        saveHistory();
        setStatus('画布已清空');
    }

    // ---------- 素描模板 ----------

    function canvasHasDrawing() {
        try {
            const w = canvas.width;
            const h = canvas.height;
            if (!w || !h) return false;
            // 采样：顶部、中心、底部三条线段上的像素，只要有非透明像素即视为有画
            const samples = [
                ctx.getImageData(0, Math.floor(h * 0.2), w, 1),
                ctx.getImageData(0, Math.floor(h * 0.5), w, 1),
                ctx.getImageData(0, Math.floor(h * 0.8), w, 1)
            ];
            for (const s of samples) {
                const d = s.data;
                for (let i = 3; i < d.length; i += 4) {
                    if (d[i] > 0) return true;
                }
            }
            return false;
        } catch (err) {
            return false;
        }
    }

    function loadTemplate(key) {
        if (!key || key === 'none') {
            return;
        }
        const svg = TEMPLATE_SVGS[key];
        if (!svg) return;

        if (canvasHasDrawing()) {
            const ok = confirm('切换素描模板会清空当前画布，继续吗？');
            if (!ok) {
                // 还原 picker 的值到当前已加载模板
                if (templatePicker) templatePicker.value = state.templateKey || 'none';
                return;
            }
        }

        const blob = new Blob([svg], { type: 'image/svg+xml' });
        const blobUrl = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
            URL.revokeObjectURL(blobUrl);
            ctx.save();
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.globalCompositeOperation = 'source-over';
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // SVG viewBox 400x300，等比居中缩放进画布
            const svgW = 400;
            const svgH = 300;
            const scale = Math.min(canvas.width / svgW, canvas.height / svgH) * 0.85;
            const drawW = svgW * scale;
            const drawH = svgH * scale;
            const offX = (canvas.width - drawW) / 2;
            const offY = (canvas.height - drawH) / 2;
            ctx.drawImage(img, offX, offY, drawW, drawH);
            ctx.restore();

            state.templateKey = key;
            // 作为新的「基态」保存到历史
            state.history = [];
            state.redoStack = [];
            saveHistory();
            setStatus('已加载素描模板 · 可在上面着色');
        };
        img.onerror = () => {
            URL.revokeObjectURL(blobUrl);
            setStatus('模板加载失败');
        };
        img.src = blobUrl;
    }

    // ---------- 工具状态 ----------

    function setTool(tool) {
        state.tool = tool;
        btnPen.classList.toggle('active', tool === 'pen');
        btnEraser.classList.toggle('active', tool === 'eraser');
        btnPen.setAttribute('aria-pressed', String(tool === 'pen'));
        btnEraser.setAttribute('aria-pressed', String(tool === 'eraser'));
    }

    function setStatus(text) {
        if (statusEl) statusEl.textContent = text;
    }

    function applyStrokeStyle(toolOverride) {
        const dpr = window.devicePixelRatio || 1;
        const t = toolOverride || state.tool;
        ctx.lineWidth = state.size * dpr;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        if (t === 'eraser') {
            ctx.globalCompositeOperation = 'destination-out';
            ctx.strokeStyle = 'rgba(0,0,0,1)';
        } else {
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeStyle = state.color;
        }
    }

    // ---------- 鼠标 / 触屏 ----------

    function pointerPos(evt) {
        const rect = canvas.getBoundingClientRect();
        const clientX = evt.clientX ?? (evt.touches && evt.touches[0] && evt.touches[0].clientX);
        const clientY = evt.clientY ?? (evt.touches && evt.touches[0] && evt.touches[0].clientY);
        const x = ((clientX - rect.left) / rect.width) * canvas.width;
        const y = ((clientY - rect.top) / rect.height) * canvas.height;
        return { x, y };
    }

    function beginStroke(evt) {
        evt.preventDefault();
        state.drawing = true;
        const { x, y } = pointerPos(evt);
        state.lastX = x;
        state.lastY = y;
        applyStrokeStyle();
        ctx.beginPath();
        ctx.arc(x, y, ctx.lineWidth / 2, 0, Math.PI * 2);
        ctx.fillStyle = state.tool === 'eraser' ? 'rgba(0,0,0,1)' : state.color;
        ctx.fill();
    }

    function moveStroke(evt) {
        if (!state.drawing) return;
        evt.preventDefault();
        const { x, y } = pointerPos(evt);
        // 手势识别循环会在 mousemove 之间 applyStrokeStyle('pen')，
        // 这会把 globalCompositeOperation 改回 source-over，导致本次
        // 鼠标橡皮擦回写成画笔。每帧重新应用本地工具样式即可。
        applyStrokeStyle();
        ctx.beginPath();
        ctx.moveTo(state.lastX, state.lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        state.lastX = x;
        state.lastY = y;
    }

    function endStroke(evt) {
        if (!state.drawing) return;
        if (evt) evt.preventDefault();
        state.drawing = false;
        ctx.globalCompositeOperation = 'source-over';
        saveHistory();
    }

    // ---------- 其它操作 ----------

    function savePng() {
        canvas.toBlob((blob) => {
            if (!blob) { setStatus('保存失败'); return; }
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            const ts = new Date().toISOString().replace(/[:.]/g, '-');
            a.href = url;
            a.download = `bindraw-${ts}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            setTimeout(() => URL.revokeObjectURL(url), 2000);
            setStatus('已下载 PNG');
        }, 'image/png');
    }

    // ---------- 手势驱动绘制 ----------

    function updateCursor(normX, normY, visible, color) {
        if (!cursorEl) return;
        if (!visible) {
            cursorEl.style.opacity = '0';
            return;
        }
        const rect = canvas.getBoundingClientRect();
        const hostRect = cursorEl.parentElement.getBoundingClientRect();
        const x = (rect.left - hostRect.left) + normX * rect.width;
        const y = (rect.top - hostRect.top) + normY * rect.height;
        cursorEl.style.left = `${x}px`;
        cursorEl.style.top = `${y}px`;
        cursorEl.style.opacity = '1';
        cursorEl.style.borderColor = color || state.color;
    }

    function smoothPos(nx, ny) {
        if (state.smoothX === null) {
            state.smoothX = nx;
            state.smoothY = ny;
        } else {
            const alpha = 0.45;
            state.smoothX = state.smoothX * (1 - alpha) + nx * alpha;
            state.smoothY = state.smoothY * (1 - alpha) + ny * alpha;
        }
        return { nx: state.smoothX, ny: state.smoothY };
    }

    function resetSmooth() {
        state.smoothX = null;
        state.smoothY = null;
    }

    function canvasCoordFromNorm(nx, ny) {
        return { x: nx * canvas.width, y: ny * canvas.height };
    }

    function gesturePenDown(nx, ny) {
        const { x, y } = canvasCoordFromNorm(nx, ny);
        state.lastX = x;
        state.lastY = y;
        state.gestureDrawing = true;
        applyStrokeStyle('pen');
        ctx.beginPath();
        ctx.arc(x, y, ctx.lineWidth / 2, 0, Math.PI * 2);
        ctx.fillStyle = state.color;
        ctx.fill();
    }

    function gesturePenMove(nx, ny) {
        const { x, y } = canvasCoordFromNorm(nx, ny);
        applyStrokeStyle('pen');
        ctx.beginPath();
        ctx.moveTo(state.lastX, state.lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        state.lastX = x;
        state.lastY = y;
    }

    function gesturePenUp() {
        if (state.gestureDrawing) {
            state.gestureDrawing = false;
            saveHistory();
        }
    }

    function onGestureFrame(payload) {
        if (!gestureToggle.checked) return;
        const { gesture, fingertip, disabled } = payload;

        if (disabled || !fingertip) {
            gesturePenUp();
            state.activeGesture = 'none';
            state.nonPointFrames = 0;
            resetSmooth();
            updateCursor(0, 0, false);
            setStatus('未检测到手');
            return;
        }

        const { nx, ny } = smoothPos(fingertip.nx, fingertip.ny);
        state.lastNormX = nx;
        state.lastNormY = ny;

        // 画画时的粘滞判定：已在画画且当前帧是 none，
        // 在 DRAW_STICKY_FRAMES 帧内仍当作 point，避免偶发误判导致断线
        let effectiveGesture = gesture;
        if (state.gestureDrawing && gesture !== 'point') {
            state.nonPointFrames += 1;
            if (state.nonPointFrames <= DRAW_STICKY_FRAMES) {
                effectiveGesture = 'point';
            }
        } else if (gesture === 'point') {
            state.nonPointFrames = 0;
        }

        if (effectiveGesture === 'point') {
            if (!state.gestureDrawing) {
                gesturePenDown(nx, ny);
            } else {
                gesturePenMove(nx, ny);
            }
            updateCursor(nx, ny, true, state.color);
            setStatus('✏️ 食指作画中');
        } else {
            gesturePenUp();
            state.nonPointFrames = 0;
            updateCursor(nx, ny, true, state.color);
            setStatus('🖐 抬笔 · 竖食指开始作画');
        }

        state.activeGesture = gesture;
    }

    // ---------- 事件绑定 ----------

    function bindEvents() {
        btnPen.addEventListener('click', () => setTool('pen'));
        btnEraser.addEventListener('click', () => setTool('eraser'));
        btnUndo.addEventListener('click', undo);
        btnRedo.addEventListener('click', redo);
        btnClear.addEventListener('click', () => {
            if (confirm('确定要清空画布吗？')) clearCanvas();
        });
        btnSave.addEventListener('click', savePng);

        colorInput.addEventListener('input', (e) => {
            state.color = e.target.value;
        });
        sizeSlider.addEventListener('input', (e) => {
            state.size = parseInt(e.target.value, 10);
            if (sizeValue) sizeValue.textContent = String(state.size);
        });

        if (templatePicker) {
            templatePicker.addEventListener('change', (e) => {
                const key = e.target.value;
                if (key === 'none') {
                    if (canvasHasDrawing() && !confirm('清空画布并移除模板？')) {
                        templatePicker.value = state.templateKey || 'none';
                        return;
                    }
                    clearCanvas();
                    state.templateKey = 'none';
                    return;
                }
                loadTemplate(key);
            });
        }

        // 鼠标
        canvas.addEventListener('mousedown', beginStroke);
        canvas.addEventListener('mousemove', moveStroke);
        window.addEventListener('mouseup', endStroke);
        canvas.addEventListener('mouseleave', endStroke);

        // 触屏
        canvas.addEventListener('touchstart', beginStroke, { passive: false });
        canvas.addEventListener('touchmove', moveStroke, { passive: false });
        canvas.addEventListener('touchend', endStroke, { passive: false });
        canvas.addEventListener('touchcancel', endStroke, { passive: false });

        // 快捷键
        window.addEventListener('keydown', (e) => {
            const isMod = e.ctrlKey || e.metaKey;
            if (!isMod) return;
            const key = e.key.toLowerCase();
            if (key === 'z' && !e.shiftKey) { e.preventDefault(); undo(); }
            else if ((key === 'z' && e.shiftKey) || key === 'y') { e.preventDefault(); redo(); }
            else if (key === 's') { e.preventDefault(); savePng(); }
        });

        // 窗口缩放
        window.addEventListener('resize', () => resizeCanvas(true));

        // 手势控制开关 → 同步到 camera
        if (gestureToggle) {
            gestureToggle.addEventListener('change', (e) => {
                if (window.BindrawCamera && typeof window.BindrawCamera.setEnabled === 'function') {
                    window.BindrawCamera.setEnabled(e.target.checked);
                }
                if (e.target.checked) {
                    document.body.classList.remove('no-camera');
                } else {
                    document.body.classList.add('no-camera');
                    gesturePenUp();
                    // EMA 状态必须清掉，否则重新开启手势时会把上一次的指尖
                    // 位置和新的第一帧混合，导致 penDown 落笔在错误的插值点
                    resetSmooth();
                    state.nonPointFrames = 0;
                    state.lastNormX = null;
                    state.lastNormY = null;
                    updateCursor(0, 0, false);
                }
            });
        }

        // 摄像头显示开关
        if (cameraToggle) {
            cameraToggle.addEventListener('change', (e) => {
                if (window.BindrawCamera && typeof window.BindrawCamera.setVisible === 'function') {
                    window.BindrawCamera.setVisible(e.target.checked);
                }
            });
        }

        // 骨架显示开关
        if (skeletonToggle) {
            skeletonToggle.addEventListener('change', (e) => {
                if (window.BindrawCamera && typeof window.BindrawCamera.setShowSkeleton === 'function') {
                    window.BindrawCamera.setShowSkeleton(e.target.checked);
                }
            });
        }

        // 手势图例折叠
        if (legendToggle && legendBody) {
            legendToggle.addEventListener('click', () => {
                const expanded = legendToggle.getAttribute('aria-expanded') === 'true';
                legendToggle.setAttribute('aria-expanded', String(!expanded));
                legendBody.hidden = expanded;
            });
        }
    }

    // ---------- 启动 ----------

    function init() {
        setTool('pen');
        resizeCanvas(false);
        saveHistory();
        bindEvents();

        if (window.BindrawCamera && typeof window.BindrawCamera.onGesture === 'function') {
            window.BindrawCamera.onGesture(onGestureFrame);
        } else {
            console.warn('[bindraw] BindrawCamera not available');
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
