const { DIMENSION_LABELS, DIMENSION_KEYS } = require('../../utils/types')

Page({
  data: {
    type: {},
    dimensions: []
  },

  onLoad() {
    const app = getApp()
    const type = app.globalData.resultType
    const levels = app.globalData.resultLevels

    if (!type) {
      wx.redirectTo({ url: '/pages/index/index' })
      return
    }

    const dimensions = DIMENSION_KEYS.map(k => {
      const level = levels[k] || 'M'
      let percent, levelText, levelClass
      if (level === 'L') {
        percent = 25; levelText = '低'; levelClass = 'low'
      } else if (level === 'M') {
        percent = 55; levelText = '中'; levelClass = 'mid'
      } else {
        percent = 85; levelText = '高'; levelClass = 'high'
      }
      return { key: k, label: DIMENSION_LABELS[k], level, percent, levelText, levelClass }
    })

    this.setData({ type, dimensions })
  },

  onShareAppMessage() {
    const t = this.data.type
    return {
      title: `我的SBTI人格是 ${t.emoji} ${t.code}（${t.name}）！你是什么？`,
      path: '/pages/index/index'
    }
  },

  onShareTimeline() {
    const t = this.data.type
    return {
      title: `我的SBTI人格是 ${t.emoji} ${t.code}（${t.name}）— 快来测测你是哪种！`
    }
  },

  onSaveImage() {
    const t = this.data.type
    const dims = this.data.dimensions

    const query = wx.createSelectorQuery()
    query.select('#shareCanvas')
      .fields({ node: true, size: true })
      .exec((res) => {
        if (!res[0]) {
          wx.showToast({ title: '生成失败', icon: 'none' })
          return
        }
        const canvas = res[0].node
        const ctx = canvas.getContext('2d')
        const dpr = wx.getWindowInfo().pixelRatio
        const w = 750
        const h = 1200
        canvas.width = w * dpr
        canvas.height = h * dpr
        ctx.scale(dpr, dpr)

        // Background
        const bgGrad = ctx.createLinearGradient(0, 0, 0, h)
        bgGrad.addColorStop(0, '#0a0a1a')
        bgGrad.addColorStop(1, '#1a1a3e')
        ctx.fillStyle = bgGrad
        ctx.fillRect(0, 0, w, h)

        // Header
        ctx.fillStyle = '#6c6c8a'
        ctx.font = '24px sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText('你的SBTI人格类型', w / 2, 80)

        // Emoji
        ctx.font = '80px sans-serif'
        ctx.fillText(t.emoji, w / 2, 200)

        // Code
        ctx.font = 'bold 56px sans-serif'
        ctx.fillStyle = '#ffffff'
        ctx.fillText(t.code, w / 2, 280)

        // Name
        ctx.font = '28px sans-serif'
        ctx.fillStyle = '#b8b8d4'
        ctx.fillText(`${t.name} · ${t.nameEn}`, w / 2, 330)

        // Tagline
        ctx.font = 'italic 22px sans-serif'
        ctx.fillStyle = '#fd79a8'
        ctx.fillText(`"${t.tagline}"`, w / 2, 380)

        // Dimensions
        ctx.textAlign = 'left'
        ctx.font = '20px sans-serif'
        const startY = 440
        const colW = 340
        dims.forEach((d, i) => {
          const col = i < 8 ? 0 : 1
          const row = i < 8 ? i : i - 8
          const x = 40 + col * colW
          const y = startY + row * 56

          ctx.fillStyle = '#b8b8d4'
          ctx.fillText(d.label, x, y)

          // Bar background
          ctx.fillStyle = 'rgba(255,255,255,0.08)'
          ctx.fillRect(x + 120, y - 14, 140, 16)

          // Bar fill
          const barGrad = ctx.createLinearGradient(x + 120, 0, x + 260, 0)
          barGrad.addColorStop(0, '#6c5ce7')
          barGrad.addColorStop(1, '#fd79a8')
          ctx.fillStyle = barGrad
          ctx.fillRect(x + 120, y - 14, 140 * d.percent / 100, 16)

          // Level text
          ctx.fillStyle = d.levelClass === 'high' ? '#fd79a8' : d.levelClass === 'low' ? '#6c5ce7' : '#a29bfe'
          ctx.textAlign = 'right'
          ctx.fillText(d.levelText, x + 290, y)
          ctx.textAlign = 'left'
        })

        // Footer
        ctx.textAlign = 'center'
        ctx.fillStyle = '#6c6c8a'
        ctx.font = '20px sans-serif'
        ctx.fillText('SBTI 沙雕大测试 — 扫码测测你的人格', w / 2, h - 40)

        wx.canvasToTempFilePath({
          canvas,
          success: (res) => {
            wx.saveImageToPhotosAlbum({
              filePath: res.tempFilePath,
              success: () => {
                wx.showToast({ title: '已保存到相册', icon: 'success' })
              },
              fail: () => {
                wx.showToast({ title: '保存失败，请授权相册权限', icon: 'none' })
              }
            })
          }
        })
      })
  },

  onRetry() {
    wx.redirectTo({ url: '/pages/quiz/quiz' })
  },

  onGoHome() {
    wx.reLaunch({ url: '/pages/index/index' })
  }
})
