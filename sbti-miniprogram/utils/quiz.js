const { DIMENSION_KEYS, SBTI_TYPES } = require('./types')

function calculateResult(dimensionScores) {
  const levels = {}
  DIMENSION_KEYS.forEach(k => {
    const s = dimensionScores[k] || 4
    if (s <= 3) levels[k] = 'L'
    else if (s <= 5) levels[k] = 'M'
    else levels[k] = 'H'
  })

  let bestType = SBTI_TYPES[0]
  let bestScore = -1

  SBTI_TYPES.forEach(t => {
    let matchScore = 0
    let totalDims = 0
    for (const dim in t.pattern) {
      totalDims++
      const expected = t.pattern[dim]
      const actual = levels[dim]
      if (actual === expected) {
        matchScore += 3
      } else if (
        (actual === 'M' && (expected === 'L' || expected === 'H')) ||
        (expected === 'M' && (actual === 'L' || actual === 'H'))
      ) {
        matchScore += 1
      }
    }
    const normalized = totalDims > 0 ? matchScore / totalDims : 0
    const finalScore = normalized + (matchScore * 0.05)
    if (finalScore > bestScore) {
      bestScore = finalScore
      bestType = t
    }
  })

  return { type: bestType, levels }
}

module.exports = { calculateResult }
