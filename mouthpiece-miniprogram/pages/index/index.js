Page({
  data: {
    types: [],
    stats: {
      questions: 30,
      models: 5,
      dimensions: 15,
      types: 25
    }
  },

  onLoad() {
    var self = this
    setTimeout(function() {
      var typesModule = require('../../utils/types')
      var types = typesModule.SBTI_TYPES.map(function(t) {
        return { code: t.code, emoji: t.emoji, name: t.name }
      })
      self.setData({ types: types })
    }, 50)
  },

  onShareAppMessage() {
    return {
      title: '互联网嘴替测试 — 测测你是哪种网络人格？',
      path: '/pages/index/index'
    }
  },

  startQuiz() {
    wx.navigateTo({ url: '/pages/quiz/quiz' })
  },

  viewType(e) {
    var code = e.currentTarget.dataset.code
    wx.navigateTo({ url: '/pages/type-detail/type-detail?code=' + code })
  }
})
