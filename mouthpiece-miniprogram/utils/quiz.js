/**
 * 互联网嘴替测试 — 评分引擎
 */

var typesModule = require('./types')
var DIMENSION_KEYS = typesModule.DIMENSION_KEYS
var SBTI_TYPES = typesModule.SBTI_TYPES

function calculateResult(dimensionScores) {
  var levels = {}
  DIMENSION_KEYS.forEach(function(k) {
    var s = dimensionScores[k] || 4
    if (s <= 3) levels[k] = 'L'
    else if (s <= 5) levels[k] = 'M'
    else levels[k] = 'H'
  })

  var bestType = SBTI_TYPES[0]
  var bestScore = -1

  SBTI_TYPES.forEach(function(t) {
    var matchScore = 0
    var totalDims = 0
    for (var dim in t.pattern) {
      totalDims++
      var expected = t.pattern[dim]
      var actual = levels[dim]
      if (actual === expected) {
        matchScore += 3
      } else if (
        (actual === 'M' && (expected === 'L' || expected === 'H')) ||
        (expected === 'M' && (actual === 'L' || actual === 'H'))
      ) {
        matchScore += 1
      }
    }
    var normalized = totalDims > 0 ? matchScore / totalDims : 0
    var finalScore = normalized + (matchScore * 0.05)
    if (finalScore > bestScore) {
      bestScore = finalScore
      bestType = t
    }
  })

  return { type: bestType, levels: levels }
}

module.exports = { calculateResult: calculateResult }
