/**
 * SBTI Test Questions
 * 30 questions: 5 models × 3 dimensions × 2 questions per dimension
 *
 * Each question maps to a dimension.
 * Each option has a score from 1-4 (low to high on that dimension).
 */

const QUESTIONS = [
  // ============ Self Model (自我模型) ============

  // selfEsteem - Q1
  {
    model: '🪞 自我模型',
    dimension: 'selfEsteem',
    text: '照镜子的时候，你通常在想什么？',
    options: [
      { text: '这人是谁？怎么看起来这么疲惫', score: 1 },
      { text: '还行吧，凑合能看', score: 2 },
      { text: '今天状态不错啊', score: 3 },
      { text: '镜子里这个人也太好看了吧', score: 4 }
    ]
  },
  // selfEsteem - Q2
  {
    model: '🪞 自我模型',
    dimension: 'selfEsteem',
    text: '被当众夸奖时，你的反应是？',
    options: [
      { text: '尴尬到脚趾抠地，想原地消失', score: 1 },
      { text: '谦虚地说"没有没有"但心里暗爽', score: 2 },
      { text: '坦然接受，微笑说谢谢', score: 3 },
      { text: '那当然了，终于有人发现我的闪光点了', score: 4 }
    ]
  },

  // selfClarity - Q3
  {
    model: '🪞 自我模型',
    dimension: 'selfClarity',
    text: '有人问你"你到底想要什么"，你会？',
    options: [
      { text: '啊？我也不知道...让我想想（然后想了三天）', score: 1 },
      { text: '大概知道，但说不太清楚', score: 2 },
      { text: '我有几个方向，正在探索', score: 3 },
      { text: '我很清楚，而且已经在行动了', score: 4 }
    ]
  },
  // selfClarity - Q4
  {
    model: '🪞 自我模型',
    dimension: 'selfClarity',
    text: '做完一个重要的选择后，你通常会？',
    options: [
      { text: '反复纠结，觉得另一个选择可能更好', score: 1 },
      { text: '偶尔会想"要是当初..."', score: 2 },
      { text: '想一下就过了，不太纠结', score: 3 },
      { text: '从不后悔，每个选择都是最好的安排', score: 4 }
    ]
  },

  // coreValue - Q5
  {
    model: '🪞 自我模型',
    dimension: 'coreValue',
    text: '如果全世界都反对你的决定，你会？',
    options: [
      { text: '立刻怀疑自己，可能真的是我错了', score: 1 },
      { text: '虽然坚持但内心动摇得厉害', score: 2 },
      { text: '听听大家的意见再决定', score: 3 },
      { text: '我的人生我做主，管你们怎么说', score: 4 }
    ]
  },
  // coreValue - Q6
  {
    model: '🪞 自我模型',
    dimension: 'coreValue',
    text: '你觉得自己在这个世界上的价值是？',
    options: [
      { text: '说实话...我也不确定我有什么价值', score: 1 },
      { text: '应该有点用吧，虽然不太突出', score: 2 },
      { text: '我在很多方面都能发挥作用', score: 3 },
      { text: '我是独一无二的，没有我世界会不一样', score: 4 }
    ]
  },

  // ============ Emotion Model (情感模型) ============

  // attachSecurity - Q7
  {
    model: '💗 情感模型',
    dimension: 'attachSecurity',
    text: '你最好的朋友突然三天没回你消息，你会？',
    options: [
      { text: '疯狂胡思乱想：是不是我做错了什么？', score: 1 },
      { text: '有点不安，频繁查看手机', score: 2 },
      { text: '他/她可能忙吧，等等看', score: 3 },
      { text: '没事，各忙各的很正常', score: 4 }
    ]
  },
  // attachSecurity - Q8
  {
    model: '💗 情感模型',
    dimension: 'attachSecurity',
    text: '在一段亲密关系中，你最担心的是？',
    options: [
      { text: '被抛弃，所以我会加倍付出来留住对方', score: 1 },
      { text: '对方不够在乎我', score: 2 },
      { text: '两个人能不能一起成长', score: 3 },
      { text: '我不太担心，顺其自然就好', score: 4 }
    ]
  },

  // emotionalInvest - Q9
  {
    model: '💗 情感模型',
    dimension: 'emotionalInvest',
    text: '看到一部特别感人的电影，你会？',
    options: [
      { text: '没什么感觉，就是个电影', score: 1 },
      { text: '有点触动，但很快就过了', score: 2 },
      { text: '默默红了眼眶', score: 3 },
      { text: '哭得稀里哗啦，还要去看第二遍', score: 4 }
    ]
  },
  // emotionalInvest - Q10
  {
    model: '💗 情感模型',
    dimension: 'emotionalInvest',
    text: '你的朋友遇到困难向你求助，你会？',
    options: [
      { text: '给个建议就行了，别太卷入', score: 1 },
      { text: '尽力帮忙，但会保持距离', score: 2 },
      { text: '全力以赴帮他/她解决', score: 3 },
      { text: '比他/她自己还着急，恨不得替他/她去', score: 4 }
    ]
  },

  // boundaries - Q11
  {
    model: '💗 情感模型',
    dimension: 'boundaries',
    text: '同事让你帮忙做一个不属于你的工作，你会？',
    options: [
      { text: '虽然心里不愿意但还是答应了', score: 1 },
      { text: '纠结一下，最后还是帮了', score: 2 },
      { text: '看情况，如果不太麻烦就帮', score: 3 },
      { text: '不好意思，这不是我的职责范围', score: 4 }
    ]
  },
  // boundaries - Q12
  {
    model: '💗 情感模型',
    dimension: 'boundaries',
    text: '你能拒绝别人吗？',
    options: [
      { text: '完全不能，我是"不"字困难户', score: 1 },
      { text: '很难，但逼急了也会说不', score: 2 },
      { text: '可以，但会找个委婉的方式', score: 3 },
      { text: '当然，拒绝是我的基本技能', score: 4 }
    ]
  },

  // ============ Attitude Model (态度模型) ============

  // worldview - Q13
  {
    model: '🌍 态度模型',
    dimension: 'worldview',
    text: '你觉得这个世界总体来说是？',
    options: [
      { text: '一坨巨大的💩，没救了', score: 1 },
      { text: '问题很多，但也不是完全没希望', score: 2 },
      { text: '有好有坏，但总体在变好', score: 3 },
      { text: '充满美好和可能性！', score: 4 }
    ]
  },
  // worldview - Q14
  {
    model: '🌍 态度模型',
    dimension: 'worldview',
    text: '看到新闻里的负面事件，你的第一反应是？',
    options: [
      { text: '果然如此，这个世界就是这样', score: 1 },
      { text: '叹口气，然后刷下一条', score: 2 },
      { text: '希望事情能好转', score: 3 },
      { text: '想想自己能做什么来帮忙', score: 4 }
    ]
  },

  // ruleFlexibility - Q15
  {
    model: '🌍 态度模型',
    dimension: 'ruleFlexibility',
    text: '交通灯变红了，但马路上一辆车都没有，你会？',
    options: [
      { text: '红灯就是红灯，必须等', score: 1 },
      { text: '等一等，确认安全了再过', score: 2 },
      { text: '看看周围没人就过了', score: 3 },
      { text: '等什么等，直接走', score: 4 }
    ]
  },
  // ruleFlexibility - Q16
  {
    model: '🌍 态度模型',
    dimension: 'ruleFlexibility',
    text: '领导制定了一个你觉得很蠢的规定，你会？',
    options: [
      { text: '规定就是规定，照做就好', score: 1 },
      { text: '私下吐槽但还是遵守', score: 2 },
      { text: '找机会跟领导提建议', score: 3 },
      { text: '直接无视或者当面怼', score: 4 }
    ]
  },

  // meaningfulness - Q17
  {
    model: '🌍 态度模型',
    dimension: 'meaningfulness',
    text: '半夜三点，你突然在想：人活着到底为了什么？',
    options: [
      { text: '不为了什么，活着就是在消耗氧气', score: 1 },
      { text: '说不清，但好像也没必要想这么深', score: 2 },
      { text: '为了体验生活中的喜怒哀乐吧', score: 3 },
      { text: '为了实现自己的梦想和价值！', score: 4 }
    ]
  },
  // meaningfulness - Q18
  {
    model: '🌍 态度模型',
    dimension: 'meaningfulness',
    text: '如果突然获得了一个月的假期，你会？',
    options: [
      { text: '躺平，什么都不想做，让我死一死', score: 1 },
      { text: '先睡三天再说', score: 2 },
      { text: '规划一些想做的事情', score: 3 },
      { text: '太好了！我有一百件想做的事！', score: 4 }
    ]
  },

  // ============ Action Model (行动模型) ============

  // motivation - Q19
  {
    model: '⚡ 行动模型',
    dimension: 'motivation',
    text: '闹钟响了，今天是工作日，你的反应是？',
    options: [
      { text: '关掉闹钟继续睡，迟到就迟到吧', score: 1 },
      { text: '在床上挣扎15分钟才起来', score: 2 },
      { text: '虽然不情愿但还是准时起床了', score: 3 },
      { text: '闹钟还没响我就醒了，今天有好多事要做！', score: 4 }
    ]
  },
  // motivation - Q20
  {
    model: '⚡ 行动模型',
    dimension: 'motivation',
    text: '面对一个有挑战但也很有意思的新项目，你会？',
    options: [
      { text: '算了吧，多一事不如少一事', score: 1 },
      { text: '观望一下再说', score: 2 },
      { text: '感兴趣，可以试试', score: 3 },
      { text: '太好了，我来我来！', score: 4 }
    ]
  },

  // decisionStyle - Q21
  {
    model: '⚡ 行动模型',
    dimension: 'decisionStyle',
    text: '菜单上有50道菜，你通常怎么选？',
    options: [
      { text: '选择困难症发作，最后让别人帮我点', score: 1 },
      { text: '纠结很久，最后随便点一个', score: 2 },
      { text: '看几个顺眼的选一个', score: 3 },
      { text: '三秒决定，吃什么不是吃', score: 4 }
    ]
  },
  // decisionStyle - Q22
  {
    model: '⚡ 行动模型',
    dimension: 'decisionStyle',
    text: '网购的时候，你通常会？',
    options: [
      { text: '比价一周，看完所有评论再下单', score: 1 },
      { text: '多看几家对比一下', score: 2 },
      { text: '看着还行就买了', score: 3 },
      { text: '看到喜欢的直接下单，不废话', score: 4 }
    ]
  },

  // executionMode - Q23
  {
    model: '⚡ 行动模型',
    dimension: 'executionMode',
    text: '你有一个很好的想法，接下来你会？',
    options: [
      { text: '想想就好了，反正也不会去做', score: 1 },
      { text: '记下来，有空的时候再说', score: 2 },
      { text: '开始制定计划，找时间执行', score: 3 },
      { text: '立刻行动！边做边想！', score: 4 }
    ]
  },
  // executionMode - Q24
  {
    model: '⚡ 行动模型',
    dimension: 'executionMode',
    text: '距离deadline还有两周，你的工作进度是？',
    options: [
      { text: '完全没开始，两周后再说', score: 1 },
      { text: '稍微看了看，不着急', score: 2 },
      { text: '已经开始了，稳步推进中', score: 3 },
      { text: '已经做完了，在改第三版了', score: 4 }
    ]
  },

  // ============ Social Model (社交模型) ============

  // socialInitiative - Q25
  {
    model: '🤝 社交模型',
    dimension: 'socialInitiative',
    text: '一个人都不认识的聚会，你会？',
    options: [
      { text: '找个角落待着，等结束', score: 1 },
      { text: '拿着手机假装很忙', score: 2 },
      { text: '找个看起来友善的人搭话', score: 3 },
      { text: '主动出击，聚会结束时跟全场混熟', score: 4 }
    ]
  },
  // socialInitiative - Q26
  {
    model: '🤝 社交模型',
    dimension: 'socialInitiative',
    text: '你的微信好友列表里有多少人是你会主动联系的？',
    options: [
      { text: '0个，我都是等别人找我', score: 1 },
      { text: '3-5个，仅限最亲近的人', score: 2 },
      { text: '十几个，经常联系的朋友', score: 3 },
      { text: '很多！我就是社交达人', score: 4 }
    ]
  },

  // interpersonalBound - Q27
  {
    model: '🤝 社交模型',
    dimension: 'interpersonalBound',
    text: '刚认识的人就想加你的所有社交账号，你会？',
    options: [
      { text: '全加，为什么不呢', score: 1 },
      { text: '加微信就行了吧', score: 2 },
      { text: '看关系远近再决定给不给', score: 3 },
      { text: '不好意思，我的社交账号是有门槛的', score: 4 }
    ]
  },
  // interpersonalBound - Q28
  {
    model: '🤝 社交模型',
    dimension: 'interpersonalBound',
    text: '你会和朋友分享你的秘密吗？',
    options: [
      { text: '所有事都会说，我没有秘密', score: 1 },
      { text: '大部分会说，除了特别私密的', score: 2 },
      { text: '只跟最信任的一两个人说', score: 3 },
      { text: '我的秘密到死都不会告诉任何人', score: 4 }
    ]
  },

  // authenticity - Q29
  {
    model: '🤝 社交模型',
    dimension: 'authenticity',
    text: '在社交场合中，你表现出来的自己和真实的自己一样吗？',
    options: [
      { text: '完全不一样，我戴了好几层面具', score: 1 },
      { text: '大部分时候会伪装一下', score: 2 },
      { text: '基本一致，偶尔客气一下', score: 3 },
      { text: '我就是我，不伪装不做作', score: 4 }
    ]
  },
  // authenticity - Q30
  {
    model: '🤝 社交模型',
    dimension: 'authenticity',
    text: '发朋友圈/社交动态的时候，你会？',
    options: [
      { text: '精心编辑，只展示最好的一面', score: 1 },
      { text: '修一修图，想一想文案', score: 2 },
      { text: '想发就发，简单配个文', score: 3 },
      { text: '直接发，真实的生活就是最好的内容', score: 4 }
    ]
  }
];
