const { QUESTIONS } = require('../../utils/questions')
const { DIMENSION_KEYS } = require('../../utils/types')
const { calculateResult } = require('../../utils/quiz')

Page({
  data: {
    currentIndex: 0,
    totalQuestions: QUESTIONS.length,
    currentQuestion: QUESTIONS[0],
    progress: (1 / QUESTIONS.length) * 100,
    letters: ['A', 'B', 'C', 'D'],
    selectedIndex: -1,
    animClass: '',
    locked: false
  },

  onLoad() {
    this.dimensionScores = {}
    DIMENSION_KEYS.forEach(k => { this.dimensionScores[k] = 0 })
  },

  onSelectOption(e) {
    if (this.data.locked) return
    const { index, score } = e.currentTarget.dataset
    const q = QUESTIONS[this.data.currentIndex]

    this.setData({ selectedIndex: parseInt(index), locked: true })

    this.dimensionScores[q.dimension] = (this.dimensionScores[q.dimension] || 0) + parseInt(score)

    setTimeout(() => {
      const nextIndex = this.data.currentIndex + 1
      if (nextIndex < QUESTIONS.length) {
        this.setData({ animClass: 'slide-out' })
        setTimeout(() => {
          this.setData({
            currentIndex: nextIndex,
            currentQuestion: QUESTIONS[nextIndex],
            progress: ((nextIndex + 1) / QUESTIONS.length) * 100,
            selectedIndex: -1,
            animClass: 'slide-in',
            locked: false
          })
          setTimeout(() => this.setData({ animClass: '' }), 300)
        }, 280)
      } else {
        this.finishQuiz()
      }
    }, 350)
  },

  finishQuiz() {
    const app = getApp()
    app.globalData.dimensionScores = this.dimensionScores

    const { type, levels } = calculateResult(this.dimensionScores)
    app.globalData.resultType = type
    app.globalData.resultLevels = levels

    wx.redirectTo({ url: '/pages/result/result' })
  }
})
