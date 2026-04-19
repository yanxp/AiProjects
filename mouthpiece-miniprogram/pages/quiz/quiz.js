Page({
  data: {
    currentQuestion: 0,
    totalQuestions: 30,
    question: null,
    progress: 0,
    letters: ['A', 'B', 'C', 'D'],
    selectedIndex: -1,
    animating: false
  },

  _questions: null,
  _dimensionKeys: null,
  _dimensionScores: null,

  onLoad() {
    var typesModule = require('../../utils/types')
    var questionsModule = require('../../utils/questions')
    this._questions = questionsModule.QUESTIONS
    this._dimensionKeys = typesModule.DIMENSION_KEYS
    this._dimensionScores = {}
    var self = this
    this._dimensionKeys.forEach(function(k) {
      self._dimensionScores[k] = 0
    })
    this.setData({ totalQuestions: this._questions.length })
    this.showQuestion(0)
  },

  showQuestion: function(index) {
    var q = this._questions[index]
    this.setData({
      currentQuestion: index,
      question: q,
      progress: ((index + 1) / this._questions.length) * 100,
      selectedIndex: -1,
      animating: false
    })
  },

  selectOption: function(e) {
    if (this.data.animating) return
    var index = e.currentTarget.dataset.index
    var score = e.currentTarget.dataset.score
    var q = this._questions[this.data.currentQuestion]

    this._dimensionScores[q.dimension] = (this._dimensionScores[q.dimension] || 0) + score

    this.setData({ selectedIndex: index, animating: true })

    var self = this
    setTimeout(function() {
      var next = self.data.currentQuestion + 1
      if (next < self._questions.length) {
        self.showQuestion(next)
      } else {
        self.showResult()
      }
    }, 400)
  },

  showResult: function() {
    var quizModule = require('../../utils/quiz')
    var result = quizModule.calculateResult(this._dimensionScores)
    var app = getApp()
    app.globalData.resultType = result.type
    app.globalData.resultLevels = result.levels
    app.globalData.dimensionScores = this._dimensionScores
    wx.redirectTo({ url: '/pages/result/result' })
  },

  goBack: function() {
    wx.navigateBack()
  }
})
