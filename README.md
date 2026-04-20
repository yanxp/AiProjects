# AiProjects

Xiaopeng Yan（严肖朋）的 AI / Web 趣味项目合集。每个子目录是一个可独立运行的项目。

## 子项目

### 1. `sbti/` — SBTI 沙雕大测试 (Web)
- 30 道题、5 种维度模型、15 个子维度、27 种人格类型
- 零依赖纯 HTML/CSS/JS，深色主题、响应式布局
- 本地预览：
  ```bash
  cd sbti && python3 -m http.server 8000
  # 浏览器访问 http://localhost:8000/
  ```

### 2. `sbti-miniprogram/` — SBTI 沙雕大测试（微信小程序）
- 原生微信小程序版本，4 个页面：`index → quiz → result → type-detail`
- Canvas 分享卡片生成、`onShareAppMessage` 支持
- 使用：用微信开发者工具导入本目录，将 `project.config.json` 里的 `appid` 替换为自己的 AppID

### 3. `mouthpiece/` — 互联网嘴替测试 (Web)
- 25 种"互联网人格"（阴阳大师、键盘侠、吃瓜群众、已读不回、摸鱼达人……）
- 5 种维度模型 × 15 个子维度 × 30 个情景化问题
- 零依赖纯 HTML/CSS/JS
- 本地预览：
  ```bash
  cd mouthpiece && python3 -m http.server 8000
  ```

### 4. `mouthpiece-miniprogram/` — 互联网嘴替测试（微信小程序）
- 原生微信小程序版本，4 个页面：`index → quiz → result → type-detail`
- `type-card` 组件承载 25 个人格网格，Canvas 分享卡片
- 使用方式同上，导入微信开发者工具并替换 AppID

### 5. `bindraw/` — 独立画板 + 可选前置摄像头手势（小朋友版）
- 打开页面即可用鼠标 / 触屏自由作画；勾选「启用手势」后才请求摄像头权限、加载 MediaPipe Hands 模型
- 手势只做极简二分类：☝️ 竖起食指 = 下笔 / 其它姿势 = 抬笔，配作画粘滞防误断
- 6 组内置动物素描线稿（猫 / 兔 / 鱼 / 蝶 / 恐龙 / 龟）给小朋友描色
- 技术栈：原生 HTML/CSS/JS + TensorFlow.js + `@tensorflow-models/hand-pose-detection` + Canvas 2D
- 本地预览（**注意需通过 HTTP 启动，`file://` 浏览器不允许摄像头**）：
  ```bash
  cd bindraw && python3 -m http.server 8000
  # 浏览器访问 http://localhost:8000/
  ```

### 6. `agent-search/` — 学术论文搜索 Agent（Python MVP）
- 面向学术场景的 Agent Search 最小骨架：`Planner → Retriever → Reader → Reflector → Synthesizer`
- 使用 LangGraph 编排状态机；OpenAlex 做论文检索；OpenAI 兼容接口调 LLM（DeepSeek / 自托管 vLLM / OpenAI 均可切换）
- 当前交付：可运行的 Python CLI demo（流式输出每一步思考 + 带编号引用的答案）；FastAPI + Next.js 完整 Web 产品的设计在 `README.md` 里预留了目录
- 技术栈：Python 3.11+ / FastAPI / LangGraph / httpx / OpenAI SDK / Pydantic
- 运行：
  ```bash
  cd agent-search/apps/api
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  cd ../..
  export LLM_BASE_URL=https://api.deepseek.com/v1
  export LLM_API_KEY=sk-xxxxxxxx
  export LLM_MODEL=deepseek-chat
  python demo.py "few-shot object detection with meta-learning"
  ```

## 目录结构

```
AiProjects/
├── sbti/                    # SBTI Web
├── sbti-miniprogram/        # SBTI 微信小程序
├── mouthpiece/              # 嘴替测试 Web
├── mouthpiece-miniprogram/  # 嘴替测试 微信小程序
├── bindraw/                 # 画板 + 可选手势识别
└── agent-search/            # 学术论文搜索 Agent (Python / LangGraph)
```

## 相关链接
- 主页：https://yanxp.github.io
- 论文 / 简历 仓库：https://github.com/yanxp/yanxp.github.io
