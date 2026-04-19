const { SBTI_TYPES } = require('../../utils/types')

Page({
  data: {
    types: SBTI_TYPES,
    scrollTags: [
      '💀 DEAD · 已死之人', '💩 SHIT · 喷子', '🐒 MALO · 猴子',
      '💸 ATM-er · 提款机', '🔥 FUCK · 野草', '🌙 SOLO · 孤儿',
      '🪩 IMSB · 自毁者', '👻 GHOST · 鬼魂', '🎮 CTRL · 拿捏者',
      '👑 BOSS · 领导者', '🤡 JOKE-R · 小丑', '💔 LOVE-R · 恋人',
      '😴 ZZZZ · 装死', '🧘 MONK · 僧人', '💃 SEXY · 万人迷',
      '😰 OH-NO · 完了人', '🙏 THAN-K · 感恩人', '😎 OJBK · 随意人'
    ]
  },

  onStartQuiz() {
    wx.navigateTo({ url: '/pages/quiz/quiz' })
  },

  onTypeCardTap(e) {
    const code = e.currentTarget.dataset.code
    wx.navigateTo({ url: `/pages/type-detail/type-detail?code=${code}` })
  },

  onShareAppMessage() {
    return {
      title: '你的SBTI沙雕人格是什么？快来测测！',
      path: '/pages/index/index'
    }
  },

  onShareTimeline() {
    return {
      title: 'SBTI沙雕大测试 — 27种人格类型，测测你是哪种？'
    }
  }
})
