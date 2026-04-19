const { SBTI_TYPES, TYPE_GROUPS } = require('../../utils/types')

Page({
  data: {
    type: null,
    groupName: ''
  },

  onLoad(options) {
    const code = options.code
    const type = SBTI_TYPES.find(t => t.code === code)
    if (!type) {
      wx.navigateBack()
      return
    }

    const group = TYPE_GROUPS[type.group]
    this.setData({
      type,
      groupName: group ? group.name : ''
    })

    wx.setNavigationBarTitle({ title: `${type.emoji} ${type.code}` })
  },

  onStartQuiz() {
    wx.navigateTo({ url: '/pages/quiz/quiz' })
  },

  onShareAppMessage() {
    const t = this.data.type
    return {
      title: `SBTI人格类型：${t.emoji} ${t.code}（${t.name}）— 你是这种人吗？`,
      path: '/pages/index/index'
    }
  }
})
