Page({
  data: {
    type: null,
    groupName: '',
    groupDesc: '',
    patternItems: []
  },

  onLoad: function(options) {
    var typesModule = require('../../utils/types')
    var SBTI_TYPES = typesModule.SBTI_TYPES
    var TYPE_GROUPS = typesModule.TYPE_GROUPS
    var DIMENSION_LABELS = typesModule.DIMENSION_LABELS

    var code = options.code
    var type = null
    for (var i = 0; i < SBTI_TYPES.length; i++) {
      if (SBTI_TYPES[i].code === code) {
        type = SBTI_TYPES[i]
        break
      }
    }
    if (!type) {
      wx.navigateBack()
      return
    }

    var group = TYPE_GROUPS[type.group] || {}
    var patternItems = []
    var keys = Object.keys(type.pattern)
    for (var j = 0; j < keys.length; j++) {
      var key = keys[j]
      var level = type.pattern[key]
      var label, cls
      if (level === 'L') { label = '低'; cls = 'low' }
      else if (level === 'M') { label = '中'; cls = 'mid' }
      else { label = '高'; cls = 'high' }
      patternItems.push({ name: DIMENSION_LABELS[key] || key, level: label, cls: cls })
    }

    this.setData({
      type: type,
      groupName: group.name || '',
      groupDesc: group.desc || '',
      patternItems: patternItems
    })

    wx.setNavigationBarTitle({ title: type.emoji + ' ' + type.name })
  },

  onShareAppMessage: function() {
    var t = this.data.type
    return {
      title: t.emoji + ' ' + t.code + ' — ' + t.name + '：' + t.tagline,
      path: '/pages/index/index'
    }
  },

  startQuiz: function() {
    wx.navigateTo({ url: '/pages/quiz/quiz' })
  }
})
