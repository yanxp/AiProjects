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

## 目录结构

```
AiProjects/
├── sbti/                    # SBTI Web
├── sbti-miniprogram/        # SBTI 微信小程序
├── mouthpiece/              # 嘴替测试 Web
└── mouthpiece-miniprogram/  # 嘴替测试 微信小程序
```

## 相关链接
- 主页：https://yanxp.github.io
- 论文 / 简历 仓库：https://github.com/yanxp/yanxp.github.io
