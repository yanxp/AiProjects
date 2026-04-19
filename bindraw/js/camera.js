/**
 * camera.js — 前置摄像头采集 + MediaPipe Hands 手势识别
 *
 * - 通过 getUserMedia 获取前置摄像头画面
 * - 使用 @tensorflow-models/hand-pose-detection 检测手部 21 个关键点
 *   优先 mediapipe runtime（更稳定），失败时回退到 tfjs runtime
 * - 基于手指伸展状态做轻量启发式手势分类
 * - 在覆盖 canvas 上绘制骨架与指尖光标
 * - 通过回调将 { gesture, fingertip } 广播给画板模块
 *
 * 导出：window.BindrawCamera = {
 *   init, captureFrame, onGesture, setEnabled, isEnabled, setShowSkeleton, setVisible
 * }
 */
(function () {
    'use strict';

    const videoEl = document.getElementById('camera-video');
    const overlayEl = document.getElementById('camera-overlay');
    const statusEl = document.getElementById('camera-status');
    const emojiEl = document.getElementById('gesture-emoji');
    const nameEl = document.getElementById('gesture-name');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingTitle = document.getElementById('loading-title');
    const loadingSub = document.getElementById('loading-sub');
    const readyBanner = document.getElementById('ready-banner');
    const readyBannerSub = document.getElementById('ready-banner-sub');
    const overlayCtx = overlayEl.getContext('2d');
    const originalTitle = document.title;

    // MediaPipe Hands 骨架连接
    const HAND_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 4],           // thumb
        [0, 5], [5, 6], [6, 7], [7, 8],           // index
        [5, 9], [9, 10], [10, 11], [11, 12],      // middle
        [9, 13], [13, 14], [14, 15], [15, 16],    // ring
        [13, 17], [17, 18], [18, 19], [19, 20],   // pinky
        [0, 17]                                    // palm base
    ];

    const GESTURE_META = {
        point: { emoji: '☝️', name: '食指作画' },
        none:  { emoji: '🖐', name: '抬笔 / 等待食指' }
    };

    // 放宽后的手指伸展阈值（tip 到 MCP 的距离 / PIP 到 MCP 的距离）
    // 之前 1.6 过于严格，很多自然姿态下的食指也被判成收起
    const FINGER_EXTEND_RATIO = 1.35;
    const THUMB_EXTEND_RATIO = 1.05;

    let detector = null;
    let stream = null;
    let running = false;
    let enabled = false;           // 默认关闭：页面加载不请求摄像头
    let initStarted = false;       // init() 是否已被调用过（避免重复请求权限）
    let initSucceeded = false;     // init() 是否已成功到能开始识别
    let showSkeleton = true;
    const listeners = [];

    // 节流诊断日志（每 ~30 帧打印一次）
    let frameCount = 0;

    function setStatus(text) {
        if (statusEl) statusEl.textContent = text;
    }

    // init 失败时把页面恢复到「无摄像头」状态：取消勾选手势开关 + 加回 body 标记
    function revertToNoCamera() {
        const toggle = document.getElementById('gesture-toggle');
        if (toggle) toggle.checked = false;
        document.body.classList.add('no-camera');
    }

    function setLoading(visible, title, sub) {
        if (!loadingOverlay) return;
        if (title && loadingTitle) loadingTitle.textContent = title;
        if (sub && loadingSub) loadingSub.textContent = sub;
        if (visible) {
            loadingOverlay.classList.remove('hidden');
        } else {
            loadingOverlay.classList.add('hidden');
        }
    }

    let readyFired = false;
    function showReadyBanner() {
        if (readyFired || !readyBanner) return;
        readyFired = true;
        readyBanner.hidden = false;
        // 强制一次回流以触发 transition
        // eslint-disable-next-line no-unused-expressions
        readyBanner.offsetHeight;
        readyBanner.classList.add('show');
        document.title = '✅ 就绪 — ' + originalTitle;
        if (readyBannerSub) {
            readyBannerSub.textContent = '竖起食指即可开始作画';
        }
        setTimeout(() => {
            readyBanner.classList.remove('show');
            setTimeout(() => { readyBanner.hidden = true; }, 400);
        }, 2800);
    }

    function updateGestureBadge(gesture) {
        const meta = GESTURE_META[gesture] || GESTURE_META.none;
        if (emojiEl) emojiEl.textContent = meta.emoji;
        if (nameEl) nameEl.textContent = meta.name;
    }

    function dist(a, b) {
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        return Math.hypot(dx, dy);
    }

    /**
     * 基于 21 个关键点判断每根手指是否伸展。
     * 约定 [thumb, index, middle, ring, pinky]，1 表示伸展。
     */
    function fingersUp(kp) {
        const thumbExtended =
            dist(kp[4], kp[2]) > dist(kp[3], kp[2]) * THUMB_EXTEND_RATIO;
        const indexExtended =
            dist(kp[8], kp[5]) > dist(kp[6], kp[5]) * FINGER_EXTEND_RATIO;
        const middleExtended =
            dist(kp[12], kp[9]) > dist(kp[10], kp[9]) * FINGER_EXTEND_RATIO;
        const ringExtended =
            dist(kp[16], kp[13]) > dist(kp[14], kp[13]) * FINGER_EXTEND_RATIO;
        const pinkyExtended =
            dist(kp[20], kp[17]) > dist(kp[18], kp[17]) * FINGER_EXTEND_RATIO;
        return [
            thumbExtended ? 1 : 0,
            indexExtended ? 1 : 0,
            middleExtended ? 1 : 0,
            ringExtended ? 1 : 0,
            pinkyExtended ? 1 : 0
        ];
    }

    /**
     * 极简分类器：只判「食指在画 / 抬笔」两种状态。
     *
     * - 食指伸展 AND（中指收起 OR 小指、无名至少 1 根收起）→ point
     *   （忽略拇指状态；允许中指轻微抬起，保证画画不断线）
     * - 其它情况 → none（抬笔 / 等待）
     */
    function classifyGesture(kp) {
        const fingers = fingersUp(kp);
        const [, i, m, r, p] = fingers;

        // 食指必须伸展
        if (!i) return { name: 'none', fingers };

        // 如果中/无名/小指「大部分」伸展（>=2 根），多半是张开手掌 / 招手，
        // 不是要画画；这时抬笔
        const others = m + r + p;
        if (others >= 2) return { name: 'none', fingers };

        return { name: 'point', fingers };
    }

    function drawSkeleton(hand) {
        if (!showSkeleton) return;
        const kp = hand.keypoints;
        overlayCtx.save();

        overlayCtx.lineWidth = Math.max(2, overlayEl.width / 320);
        overlayCtx.strokeStyle = 'rgba(34, 197, 94, 0.95)';
        overlayCtx.beginPath();
        HAND_CONNECTIONS.forEach(([a, b]) => {
            overlayCtx.moveTo(kp[a].x, kp[a].y);
            overlayCtx.lineTo(kp[b].x, kp[b].y);
        });
        overlayCtx.stroke();

        const r = Math.max(3, overlayEl.width / 240);
        overlayCtx.fillStyle = 'rgba(37, 99, 235, 0.95)';
        kp.forEach((pt) => {
            overlayCtx.beginPath();
            overlayCtx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
            overlayCtx.fill();
        });

        // 食指指尖高亮
        overlayCtx.fillStyle = '#f97316';
        overlayCtx.beginPath();
        overlayCtx.arc(kp[8].x, kp[8].y, r * 1.8, 0, Math.PI * 2);
        overlayCtx.fill();

        overlayCtx.restore();
    }

    function syncOverlaySize() {
        if (!videoEl.videoWidth || !videoEl.videoHeight) return;
        if (
            overlayEl.width !== videoEl.videoWidth ||
            overlayEl.height !== videoEl.videoHeight
        ) {
            overlayEl.width = videoEl.videoWidth;
            overlayEl.height = videoEl.videoHeight;
        }
    }

    function notifyListeners(payload) {
        listeners.forEach((fn) => {
            try { fn(payload); } catch (err) { console.error(err); }
        });
    }

    async function detectLoop() {
        if (!detector || !running) return;
        if (!enabled) {
            requestAnimationFrame(detectLoop);
            return;
        }
        try {
            if (videoEl.readyState >= 2 && videoEl.videoWidth > 0) {
                const hands = await detector.estimateHands(videoEl, { flipHorizontal: false });
                syncOverlaySize();
                overlayCtx.clearRect(0, 0, overlayEl.width, overlayEl.height);

                if (hands && hands.length > 0) {
                    const hand = hands[0];
                    const { name: gesture, fingers } = classifyGesture(hand.keypoints);
                    drawSkeleton(hand);
                    updateGestureBadge(gesture);
                    setStatus(`识别中 · ${gesture}`);

                    const tip = hand.keypoints[8];
                    const nx = 1 - tip.x / overlayEl.width; // 水平镜像，匹配 CSS scaleX(-1)
                    const ny = tip.y / overlayEl.height;

                    if (frameCount % 30 === 0) {
                        console.debug('[bindraw] fingers =', fingers.join(''), 'gesture =', gesture);
                    }
                    frameCount += 1;

                    notifyListeners({ gesture, fingertip: { nx, ny }, raw: hand });
                } else {
                    updateGestureBadge('none');
                    setStatus('识别中 · 未检测到手');
                    notifyListeners({ gesture: 'none', fingertip: null, raw: null });
                }
            }
        } catch (err) {
            console.warn('[bindraw] detect error', err);
        }
        requestAnimationFrame(detectLoop);
    }

    async function initCamera() {
        if (initStarted) return;
        initStarted = true;
        enabled = true;

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setStatus('当前浏览器不支持摄像头 API');
            setLoading(true, '当前浏览器不支持摄像头 API', '请在最新 Chrome / Safari / Edge 中打开');
            setTimeout(() => setLoading(false), 2500);
            initStarted = false;
            enabled = false;
            revertToNoCamera();
            return;
        }

        setLoading(true, '请求摄像头权限…', '请在浏览器提示中选择「允许」');
        setStatus('请求摄像头权限…');
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });
        } catch (err) {
            console.error('[bindraw] getUserMedia failed', err);
            let message = '无法访问摄像头：' + (err && err.name ? err.name : 'UnknownError');
            let sub = '已自动关闭手势，可继续用鼠标 / 触屏作画';
            if (err && err.name === 'NotAllowedError') {
                message = '摄像头权限被拒绝';
                sub = '请在浏览器地址栏允许摄像头后再开启「启用手势」';
            } else if (err && err.name === 'NotFoundError') {
                message = '未检测到可用摄像头';
                sub = '请确认摄像头已连接并未被其它应用占用';
            } else if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
                message = '请在 HTTPS 或 localhost 下访问';
                sub = '浏览器仅在安全上下文下允许访问摄像头';
            }
            setStatus(message);
            setLoading(true, message, sub);
            // 2.5s 后自动隐藏遮罩，让用户回到画布继续作画
            setTimeout(() => setLoading(false), 2500);
            initStarted = false;
            enabled = false;
            revertToNoCamera();
            return;
        }

        videoEl.srcObject = stream;
        await new Promise((resolve) => {
            if (videoEl.readyState >= 1) return resolve();
            videoEl.addEventListener('loadedmetadata', resolve, { once: true });
        });
        try {
            await videoEl.play();
        } catch (err) {
            console.warn('[bindraw] video.play() rejected', err);
        }
        syncOverlaySize();

        setLoading(true, '加载手势识别模型…', '首屏约需下载 2–5 MB，通常 2–10 秒');
        await loadDetector();

        if (!detector) {
            setTimeout(() => setLoading(false), 2500);
            initStarted = false;
            enabled = false;
            if (stream) {
                stream.getTracks().forEach((t) => t.stop());
                stream = null;
            }
            revertToNoCamera();
            return;
        }

        setStatus('✅ 就绪 · 请把手伸到摄像头前');
        setLoading(false);
        showReadyBanner();
        initSucceeded = true;
        running = true;
        detectLoop();
    }

    async function loadDetector() {
        if (!window.handPoseDetection) {
            setStatus('hand-pose-detection 未加载，检查网络');
            setLoading(true, '手势识别脚本加载失败', '请检查网络后刷新页面');
            console.error('[bindraw] handPoseDetection global missing');
            return;
        }
        const hpd = window.handPoseDetection;

        // 先试 mediapipe runtime（更稳定、更快），失败回退 tfjs
        const attempts = [
            {
                name: 'mediapipe',
                config: {
                    runtime: 'mediapipe',
                    modelType: 'full',
                    maxHands: 1,
                    solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands'
                }
            },
            {
                name: 'tfjs(lite)',
                config: { runtime: 'tfjs', modelType: 'lite', maxHands: 1 }
            },
            {
                name: 'tfjs(full)',
                config: { runtime: 'tfjs', modelType: 'full', maxHands: 1 }
            }
        ];

        for (const attempt of attempts) {
            setStatus(`加载手势识别模型（${attempt.name}）…`);
            setLoading(true, `加载手势识别模型（${attempt.name}）…`, '首屏约需下载 2–5 MB，通常 2–10 秒');
            try {
                detector = await hpd.createDetector(
                    hpd.SupportedModels.MediaPipeHands,
                    attempt.config
                );
                console.info(`[bindraw] detector loaded via ${attempt.name}`);
                return;
            } catch (err) {
                console.warn(`[bindraw] detector ${attempt.name} failed`, err);
            }
        }
        detector = null;
        setStatus('手势模型加载失败，请检查网络后刷新');
        setLoading(true, '手势模型加载失败', '请检查网络后刷新页面');
    }

    /**
     * 截取当前视频帧（已把镜像翻回），返回离屏 canvas。
     */
    function captureFrame() {
        if (!videoEl.videoWidth || !videoEl.videoHeight) return null;
        const w = videoEl.videoWidth;
        const h = videoEl.videoHeight;
        const off = document.createElement('canvas');
        off.width = w;
        off.height = h;
        const ctx = off.getContext('2d');
        ctx.save();
        ctx.translate(w, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(videoEl, 0, 0, w, h);
        ctx.restore();
        return { canvas: off, width: w, height: h };
    }

    function onGesture(fn) {
        if (typeof fn === 'function') listeners.push(fn);
    }

    function setEnabled(flag) {
        const target = !!flag;
        if (target && !initStarted) {
            // 首次开启：惰性请求摄像头权限 + 加载模型
            initCamera();
            return;
        }
        enabled = target;
        if (!enabled) {
            overlayCtx.clearRect(0, 0, overlayEl.width, overlayEl.height);
            updateGestureBadge('none');
            notifyListeners({ gesture: 'none', fingertip: null, raw: null, disabled: true });
            setStatus('手势控制已关闭');
        } else {
            setStatus(initSucceeded ? '识别中' : '手势模型加载中…');
        }
    }

    function setShowSkeleton(flag) {
        showSkeleton = !!flag;
        if (!showSkeleton) {
            overlayCtx.clearRect(0, 0, overlayEl.width, overlayEl.height);
        }
    }

    function setVisible(flag) {
        videoEl.style.visibility = flag ? 'visible' : 'hidden';
        overlayEl.style.visibility = flag ? 'visible' : 'hidden';
    }

    function isEnabled() {
        return enabled;
    }

    window.BindrawCamera = {
        init: initCamera,
        captureFrame,
        onGesture,
        setEnabled,
        isEnabled,
        setShowSkeleton,
        setVisible
    };

    // 不自动初始化：只有用户勾选「启用手势」时才请求摄像头权限 + 加载模型。
    // 入口由 bindraw.js 中的 gesture-toggle 触发 BindrawCamera.setEnabled(true)。
})();
