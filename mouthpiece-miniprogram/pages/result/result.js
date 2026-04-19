Page({
  data: {
    type: null,
    levels: null,
    dimensions: [],
    groupName: '',
    groupDesc: ''
  },

  onLoad: function() {
    var app = getApp()
    var type = app.globalData.resultType
    var levels = app.globalData.resultLevels

    if (!type || !levels) {
      wx.redirectTo({ url: '/pages/index/index' })
      return
    }

    var typesModule = require('../../utils/types')
    var DIMENSION_LABELS = typesModule.DIMENSION_LABELS
    var DIMENSION_KEYS = typesModule.DIMENSION_KEYS
    var TYPE_GROUPS = typesModule.TYPE_GROUPS

    var dimensions = DIMENSION_KEYS.map(function(k) {
      var lv = levels[k]
      var pct, label, cls
      if (lv === 'L') { pct = 25; label = '低'; cls = 'low' }
      else if (lv === 'M') { pct = 55; label = '中'; cls = 'mid' }
      else { pct = 85; label = '高'; cls = 'high' }
      return { key: k, name: DIMENSION_LABELS[k], pct: pct, label: label, cls: cls }
    })

    var group = TYPE_GROUPS[type.group] || {}

    this.setData({
      type: type,
      levels: levels,
      dimensions: dimensions,
      groupName: group.name || '',
      groupDesc: group.desc || ''
    })
  },

  onShareAppMessage: function() {
    var t = this.data.type
    return {
      title: '我的网络人格是 ' + t.emoji + ' ' + t.code + '（' + t.name + '）！',
      path: '/pages/index/index'
    }
  },

  shareResult: function() {
    this.drawShareCard()
  },

  drawShareCard: function() {
    var t = this.data.type
    var self = this
    var query = wx.createSelectorQuery()
    query.select('#shareCanvas')
      .fields({ node: true, size: true })
      .exec(function(res) {
        if (!res[0]) {
          wx.showToast({ title: '生成失败', icon: 'none' })
          return
        }
        var canvas = res[0].node
        var ctx = canvas.getContext('2d')
        var dpr = wx.getWindowInfo().pixelRatio
        canvas.width = 600 * dpr
        canvas.height = 800 * dpr
        ctx.scale(dpr, dpr)

        // Background
        var bgGrad = ctx.createLinearGradient(0, 0, 600, 800)
        bgGrad.addColorStop(0, '#03071e')
        bgGrad.addColorStop(0.5, '#0a1128')
        bgGrad.addColorStop(1, '#03071e')
        ctx.fillStyle = bgGrad
        ctx.fillRect(0, 0, 600, 800)

        // Top accent line
        var lineGrad = ctx.createLinearGradient(0, 0, 600, 0)
        lineGrad.addColorStop(0, '#00b4d8')
        lineGrad.addColorStop(0.5, '#7209b7')
        lineGrad.addColorStop(1, '#f72585')
        ctx.fillStyle = lineGrad
        ctx.fillRect(0, 0, 600, 6)

        // Badge
        ctx.fillStyle = 'rgba(0,180,216,0.15)'
        self.roundRect(ctx, 200, 40, 200, 36, 18)
        ctx.fill()
        ctx.fillStyle = '#00b4d8'
        ctx.font = '14px sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText('互联网嘴替测试', 300, 63)

        // Emoji
        ctx.font = '72px sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText(t.emoji, 300, 160)

        // Code
        ctx.fillStyle = '#00b4d8'
        ctx.font = 'bold 28px sans-serif'
        ctx.fillText(t.code, 300, 210)

        // Name
        ctx.fillStyle = '#ffffff'
        ctx.font = 'bold 22px sans-serif'
        ctx.fillText(t.name + ' · ' + t.nameEn, 300, 248)

        // Tagline
        ctx.fillStyle = 'rgba(224,224,240,0.6)'
        ctx.font = '16px sans-serif'
        ctx.fillText('"' + t.tagline + '"', 300, 285)

        // Divider
        ctx.strokeStyle = 'rgba(255,255,255,0.1)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(60, 310)
        ctx.lineTo(540, 310)
        ctx.stroke()

        // Desc
        ctx.fillStyle = 'rgba(224,224,240,0.7)'
        ctx.font = '14px sans-serif'
        ctx.textAlign = 'left'
        self.wrapText(ctx, t.desc, 60, 345, 480, 22)

        // Traits
        ctx.textAlign = 'left'
        var tx = 60
        var ty = 520
        t.traits.forEach(function(trait) {
          var tw = ctx.measureText(trait).width + 24
          ctx.fillStyle = 'rgba(0,180,216,0.15)'
          self.roundRect(ctx, tx, ty - 14, tw, 28, 14)
          ctx.fill()
          ctx.fillStyle = '#00b4d8'
          ctx.font = '13px sans-serif'
          ctx.fillText(trait, tx + 12, ty + 4)
          tx += tw + 12
        })

        // Dimensions (top 5)
        var dims = self.data.dimensions.slice(0, 5)
        dims.forEach(function(d, i) {
          var dy = 580 + i * 36
          ctx.fillStyle = 'rgba(224,224,240,0.6)'
          ctx.font = '13px sans-serif'
          ctx.textAlign = 'left'
          ctx.fillText(d.name, 60, dy + 4)

          ctx.fillStyle = 'rgba(255,255,255,0.08)'
          self.roundRect(ctx, 180, dy - 8, 300, 16, 8)
          ctx.fill()

          var barGrad = ctx.createLinearGradient(180, 0, 180 + 300 * d.pct / 100, 0)
          barGrad.addColorStop(0, '#00b4d8')
          barGrad.addColorStop(1, '#f72585')
          ctx.fillStyle = barGrad
          self.roundRect(ctx, 180, dy - 8, 300 * d.pct / 100, 16, 8)
          ctx.fill()

          ctx.fillStyle = d.cls === 'low' ? '#4cc9f0' : d.cls === 'mid' ? '#f9c74f' : '#f72585'
          ctx.textAlign = 'right'
          ctx.fillText(d.label, 540, dy + 4)
        })

        // Footer
        ctx.fillStyle = 'rgba(224,224,240,0.3)'
        ctx.font = '12px sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText('扫码测测你是哪种网络人格 →', 300, 775)

        // Save
        setTimeout(function() {
          wx.canvasToTempFilePath({
            canvas: canvas,
            success: function(res) {
              wx.saveImageToPhotosAlbum({
                filePath: res.tempFilePath,
                success: function() { wx.showToast({ title: '已保存到相册', icon: 'success' }) },
                fail: function() { wx.showToast({ title: '保存失败', icon: 'none' }) }
              })
            },
            fail: function() { wx.showToast({ title: '生成失败', icon: 'none' }) }
          })
        }, 200)
      })
  },

  roundRect: function(ctx, x, y, w, h, r) {
    ctx.beginPath()
    ctx.moveTo(x + r, y)
    ctx.lineTo(x + w - r, y)
    ctx.arcTo(x + w, y, x + w, y + r, r)
    ctx.lineTo(x + w, y + h - r)
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r)
    ctx.lineTo(x + r, y + h)
    ctx.arcTo(x, y + h, x, y + h - r, r)
    ctx.lineTo(x, y + r)
    ctx.arcTo(x, y, x + r, y, r)
    ctx.closePath()
  },

  wrapText: function(ctx, text, x, y, maxWidth, lineHeight) {
    var line = ''
    for (var i = 0; i < text.length; i++) {
      var ch = text[i]
      var testLine = line + ch
      if (ctx.measureText(testLine).width > maxWidth) {
        ctx.fillText(line, x, y)
        line = ch
        y += lineHeight
      } else {
        line = testLine
      }
    }
    ctx.fillText(line, x, y)
  },

  retakeQuiz: function() {
    wx.redirectTo({ url: '/pages/quiz/quiz' })
  },

  goHome: function() {
    wx.redirectTo({ url: '/pages/index/index' })
  }
})
