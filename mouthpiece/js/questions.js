/**
 * 互联网嘴替测试 — 30道题目
 * 5大模型 × 3个维度 × 2题/维度
 */

const QUESTIONS = [
  // ============ 表达模型 (Expression Model) ============
  { model: '🗣️ 表达模型', dimension: 'expressionStyle', text: '朋友发了一条明显在凡尔赛的朋友圈，你会？', options: [
    { text: '默默滑过，当没看见', score: 1 },
    { text: '点个赞，不评论', score: 2 },
    { text: '评论"哇好棒！"（内心：呵呵）', score: 3 },
    { text: '直接评论"凡尔赛文学大赏第一名"', score: 4 }
  ]},
  { model: '🗣️ 表达模型', dimension: 'expressionStyle', text: '群里有人发了一个你觉得很蠢的观点，你会？', options: [
    { text: '不想回，懒得理', score: 1 },
    { text: '默默退出聊天界面', score: 2 },
    { text: '发个"..."或者"？"', score: 3 },
    { text: '打一大段话逐条反驳', score: 4 }
  ]},
  { model: '🗣️ 表达模型', dimension: 'sarcasmLevel', text: '同事说"周末加班好充实啊"，你的反应是？', options: [
    { text: '微笑点头', score: 1 },
    { text: '心里吐槽但不说出来', score: 2 },
    { text: '"是啊，充实到头发都充实掉了"', score: 3 },
    { text: '"确实，建议你一周七天都来感受这份充实"', score: 4 }
  ]},
  { model: '🗣️ 表达模型', dimension: 'sarcasmLevel', text: '老板发了一条"感恩平台"的朋友圈，你会？', options: [
    { text: '真诚点赞', score: 1 },
    { text: '装没看到', score: 2 },
    { text: '截图发给同事私聊吐槽', score: 3 },
    { text: '评论"格局！大格局！"（阴阳怪气）', score: 4 }
  ]},
  { model: '🗣️ 表达模型', dimension: 'emojiDependency', text: '你发消息不加表情包会怎样？', options: [
    { text: '完全没问题，我很少用表情包', score: 1 },
    { text: '偶尔会用，但不依赖', score: 2 },
    { text: '感觉语气会变得很严肃，必须加', score: 3 },
    { text: '不加表情包=生气，这是互联网基本法', score: 4 }
  ]},
  { model: '🗣️ 表达模型', dimension: 'emojiDependency', text: '你手机里存了多少表情包？', options: [
    { text: '不到10个，够用就行', score: 1 },
    { text: '几十个，常用的那些', score: 2 },
    { text: '几百个，按场景分类', score: 3 },
    { text: '上千个，我的表情包库比相册都大', score: 4 }
  ]},

  // ============ 战斗模型 (Battle Model) ============
  { model: '🔥 战斗模型', dimension: 'debateDesire', text: '看到评论区有人说了你不认同的观点，你会？', options: [
    { text: '关掉页面，眼不见为净', score: 1 },
    { text: '看看就好，不值得浪费时间', score: 2 },
    { text: '忍不住回一条纠正一下', score: 3 },
    { text: '开启长篇辩论模式，不赢不走', score: 4 }
  ]},
  { model: '🔥 战斗模型', dimension: 'debateDesire', text: '网上有人@你说"你这个观点不对"，你的反应是？', options: [
    { text: '可能他说得对吧，不回了', score: 1 },
    { text: '随便回一句敷衍过去', score: 2 },
    { text: '认真回复解释我的观点', score: 3 },
    { text: '先截图保存证据，然后开始逐条反驳', score: 4 }
  ]},
  { model: '🔥 战斗模型', dimension: 'opinionStrength', text: '朋友推荐了一部你觉得很难看的电影，你会？', options: [
    { text: '"还行吧"（不想得罪人）', score: 1 },
    { text: '"不太是我的菜"', score: 2 },
    { text: '诚实地说我觉得哪里不好', score: 3 },
    { text: '发一篇500字影评解释为什么是烂片', score: 4 }
  ]},
  { model: '🔥 战斗模型', dimension: 'opinionStrength', text: '有个热门话题大家都在支持一方，但你持相反观点，你会？', options: [
    { text: '算了，跟大家保持一致吧', score: 1 },
    { text: '心里不同意但不说', score: 2 },
    { text: '小心翼翼地表达不同看法', score: 3 },
    { text: '我就是要当少数派，真理不在多数人手里', score: 4 }
  ]},
  { model: '🔥 战斗模型', dimension: 'triggerThreshold', text: '有人在你的评论区留了一条阴阳怪气的话，你会？', options: [
    { text: '心态稳定，完全不受影响', score: 4 },
    { text: '有点不舒服，但很快就忘了', score: 3 },
    { text: '纠结半天要不要回怼', score: 2 },
    { text: '立刻破防，手速280开始回击', score: 1 }
  ]},
  { model: '🔥 战斗模型', dimension: 'triggerThreshold', text: '你发了一条认真的观点，结果被人嘲笑了，你会？', options: [
    { text: '不在意，每个人想法不同很正常', score: 4 },
    { text: '有点不爽，但选择无视', score: 3 },
    { text: '删掉那条动态，有点后悔发了', score: 2 },
    { text: '气到发抖，连发三条反击', score: 1 }
  ]},

  // ============ 围观模型 (Spectator Model) ============
  { model: '👀 围观模型', dimension: 'melonEating', text: '微博热搜有个明星塌房了，你会？', options: [
    { text: '不关心，继续做自己的事', score: 1 },
    { text: '瞄一眼标题就划走了', score: 2 },
    { text: '点进去看看怎么回事', score: 3 },
    { text: '翻完所有相关微博+评论区+超话+豆瓣小组', score: 4 }
  ]},
  { model: '👀 围观模型', dimension: 'melonEating', text: '同事A和同事B闹矛盾了，你的反应是？', options: [
    { text: '跟我没关系，不想参与', score: 1 },
    { text: '知道就行了，不八卦', score: 2 },
    { text: '在工作群潜水收集信息', score: 3 },
    { text: '分别找两边打听来龙去脉，然后跟其他同事分析', score: 4 }
  ]},
  { model: '👀 围观模型', dimension: 'shareDesire', text: '你看到一个超级好笑的视频，你会？', options: [
    { text: '自己笑笑就好了', score: 1 },
    { text: '可能发给最好的朋友', score: 2 },
    { text: '转发到几个群里', score: 3 },
    { text: '发朋友圈+转发所有群+私聊发给十个人', score: 4 }
  ]},
  { model: '👀 围观模型', dimension: 'shareDesire', text: '你发现了一家超好吃的餐厅，你会？', options: [
    { text: '自己偷偷享用，好吃的东西要藏起来', score: 1 },
    { text: '跟几个好朋友分享', score: 2 },
    { text: '发个朋友圈推荐一下', score: 3 },
    { text: '写一篇小红书/大众点评详细测评', score: 4 }
  ]},
  { model: '👀 围观模型', dimension: 'commentStyle', text: '你在评论区的存在感有多强？', options: [
    { text: '我从来不评论', score: 1 },
    { text: '偶尔评论，一句话解决', score: 2 },
    { text: '经常评论，会认真写', score: 3 },
    { text: '评论区就是我的主场，我的评论经常被顶到最高', score: 4 }
  ]},
  { model: '👀 围观模型', dimension: 'commentStyle', text: '你给别人点赞的频率是？', options: [
    { text: '几乎不点，我的赞很珍贵', score: 1 },
    { text: '偶尔点，看心情', score: 2 },
    { text: '经常点，看到不错的就点', score: 3 },
    { text: '见一个点一个，我是点赞永动机', score: 4 }
  ]},

  // ============ 人设模型 (Persona Model) ============
  { model: '🎭 人设模型', dimension: 'onlinePersona', text: '你在网上的形象和现实中差距大吗？', options: [
    { text: '完全一致，我在哪都一样', score: 1 },
    { text: '差不多，可能网上稍微活泼点', score: 2 },
    { text: '差距不小，网上更敢说话', score: 3 },
    { text: '简直是两个人，网上的我是理想中的我', score: 4 }
  ]},
  { model: '🎭 人设模型', dimension: 'onlinePersona', text: '如果网友要线下面基，你的反应是？', options: [
    { text: '没问题啊，我线上线下都一样', score: 1 },
    { text: '可以，但需要适应一下', score: 2 },
    { text: '有点紧张，怕他们失望', score: 3 },
    { text: '绝对不行！见面就穿帮了', score: 4 }
  ]},
  { model: '🎭 人设模型', dimension: 'socialMask', text: '发朋友圈前你会花多长时间编辑？', options: [
    { text: '想发就发，一分钟搞定', score: 1 },
    { text: '简单看看，两三分钟', score: 2 },
    { text: '精心修图+想文案，十几分钟', score: 3 },
    { text: '从拍照到修图到文案到发布，一个小时起步', score: 4 }
  ]},
  { model: '🎭 人设模型', dimension: 'socialMask', text: '你会删除之前发的朋友圈/动态吗？', options: [
    { text: '不会，发了就发了', score: 1 },
    { text: '很少删，除非实在太尴尬', score: 2 },
    { text: '会定期清理，保持形象', score: 3 },
    { text: '经常删，我的朋友圈是精心策展的画廊', score: 4 }
  ]},
  { model: '🎭 人设模型', dimension: 'attentionNeed', text: '你发了一条自认为很棒的动态，半小时没人点赞，你会？', options: [
    { text: '无所谓，发给自己看的', score: 1 },
    { text: '不太在意，继续做别的事', score: 2 },
    { text: '有点失落，刷新看看', score: 3 },
    { text: '焦虑到不行，开始考虑删掉', score: 4 }
  ]},
  { model: '🎭 人设模型', dimension: 'attentionNeed', text: '你觉得发动态最大的动力是什么？', options: [
    { text: '记录生活，给自己看', score: 1 },
    { text: '分享给关心的朋友', score: 2 },
    { text: '希望得到大家的认可和点赞', score: 3 },
    { text: '如果没人看到，那发了有什么意义？', score: 4 }
  ]},

  // ============ 生态模型 (Ecosystem Model) ============
  { model: '📱 生态模型', dimension: 'platformLoyalty', text: '你每天主要活跃在几个社交平台上？', options: [
    { text: '1-2个，够用就行', score: 1 },
    { text: '3-4个，各有各的用处', score: 2 },
    { text: '5-7个，哪里热闹去哪里', score: 3 },
    { text: '数不清，新出的App我都要试试', score: 4 }
  ]},
  { model: '📱 生态模型', dimension: 'platformLoyalty', text: '一个你常用的社交平台宣布关闭，你的反应是？', options: [
    { text: '无所谓，换一个就好', score: 4 },
    { text: '有点可惜，但很快会找到替代品', score: 3 },
    { text: '难过，那上面有很多回忆', score: 2 },
    { text: '天塌了，我的青春/社交圈/创作全在那里', score: 1 }
  ]},
  { model: '📱 生态模型', dimension: 'contentCreation', text: '你在社交平台上主要是什么角色？', options: [
    { text: '纯消费者，只看不发', score: 1 },
    { text: '偶尔发点东西，以转发为主', score: 2 },
    { text: '经常发原创内容', score: 3 },
    { text: '我就是内容本身，创作是我的日常', score: 4 }
  ]},
  { model: '📱 生态模型', dimension: 'contentCreation', text: '你觉得自己有成为KOL/博主的潜力吗？', options: [
    { text: '完全没有，我连发朋友圈都不知道写什么', score: 1 },
    { text: '可能有，但我懒得经营', score: 2 },
    { text: '有一些想法，正在尝试', score: 3 },
    { text: '当然！给我一个账号我就能起飞', score: 4 }
  ]},
  { model: '📱 生态模型', dimension: 'digitalDetox', text: '让你一天不看手机，你能做到吗？', options: [
    { text: '完全可以，很享受不被打扰的时光', score: 4 },
    { text: '应该可以，虽然会有点不习惯', score: 3 },
    { text: '很难，会忍不住偷看', score: 2 },
    { text: '绝对不可能，手机就是我的氧气', score: 1 }
  ]},
  { model: '📱 生态模型', dimension: 'digitalDetox', text: '睡前最后一件事和醒来第一件事是什么？', options: [
    { text: '睡前看书/冥想，醒来先洗漱', score: 4 },
    { text: '睡前可能会看下手机，醒来先起床', score: 3 },
    { text: '睡前刷手机到困，醒来先看消息', score: 2 },
    { text: '睡前刷到凌晨三点，醒来第一件事还是拿手机', score: 1 }
  ]}
]
