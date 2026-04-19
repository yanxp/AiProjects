/**
 * 互联网嘴替测试 — 网络人格类型定义
 * 5大模型 × 15个维度 × 25种人格类型
 */

var DIMENSION_LABELS = {
  expressionStyle:   '表达风格',
  sarcasmLevel:      '阴阳浓度',
  emojiDependency:   '表情包依赖',
  debateDesire:      '对线欲望',
  opinionStrength:   '观点强度',
  triggerThreshold:  '破防阈值',
  melonEating:       '吃瓜热情',
  shareDesire:       '转发欲望',
  commentStyle:      '评论风格',
  onlinePersona:     '网络人设',
  socialMask:        '社交面具',
  attentionNeed:     '关注需求',
  platformLoyalty:   '平台忠诚',
  contentCreation:   '内容创作',
  digitalDetox:      '戒网能力'
}

var DIMENSION_KEYS = Object.keys(DIMENSION_LABELS)

var SBTI_TYPES = [
  {
    code: 'YIN-YANG', emoji: '🗡️', name: '阴阳大师', nameEn: 'The Shade Master',
    tagline: '我没说你不好，我是说在座的各位都是垃圾。',
    desc: '你是互联网阴阳怪气的天花板。每一句话都看起来人畜无害，但懂的人自然懂。你从不直接骂人，因为你的杀伤力全在言外之意里。老板说"辛苦了"你回"不辛苦，钱给够了就不辛苦"。朋友圈三天可见？不，你的朋友圈是"阴阳三百年可见"。',
    traits: ['话里有话', '杀人不见血', '反讽高手', '懂的都懂'],
    pattern: { sarcasmLevel:'H', expressionStyle:'H', opinionStrength:'H', socialMask:'H', triggerThreshold:'H' },
    group: 'attack'
  },
  {
    code: 'ZHENG', emoji: '☀️', name: '正能量战士', nameEn: 'The Positivity Warrior',
    tagline: '加油！你是最棒的！明天会更好！',
    desc: '你是互联网正能量的最后一道防线。不管别人发什么丧的内容，你都能在评论区播撒阳光。"别难过！""你已经很棒了！""一切都会好的！"是你的三连击。有人觉得你是真的善良，有人觉得你是"何不食肉糜"——但你不在乎，因为你正在忙着给下一个陌生人点赞。',
    traits: ['正能量输出', '点赞狂魔', '鼓励达人', '永远积极'],
    pattern: { expressionStyle:'H', sarcasmLevel:'L', debateDesire:'L', commentStyle:'H', attentionNeed:'M' },
    group: 'warm'
  },
  {
    code: 'MELON', emoji: '🍉', name: '吃瓜群众', nameEn: 'The Melon Eater',
    tagline: '前排占座，瓜呢瓜呢？',
    desc: '互联网的任何八卦都逃不过你的法眼。明星塌房你第一个知道，公司内斗你门儿清，朋友圈谁删了谁你都有记录。你不生产瓜，你只是瓜的搬运工。你的浏览器收藏夹里全是各种吃瓜论坛，你的微信群名叫"前线瓜田"。',
    traits: ['八卦雷达', '信息通', '前排专业户', '不站队只看戏'],
    pattern: { melonEating:'H', shareDesire:'H', debateDesire:'L', opinionStrength:'L', commentStyle:'M' },
    group: 'spectator'
  },
  {
    code: 'KEY-B', emoji: '⌨️', name: '键盘侠', nameEn: 'The Keyboard Warrior',
    tagline: '你说得对，但是我不同意。',
    desc: '你是评论区的战神，任何话题都能被你拉进辩论场。你不是为了赢，你是为了让对方知道他错了（虽然有时候错的是你）。你的打字速度比说话快，因为线下吵架你可能吵不赢——但在网上，你就是无敌的存在。',
    traits: ['评论区战神', '对线成瘾', '打字如飞', '线上猛虎'],
    pattern: { debateDesire:'H', opinionStrength:'H', expressionStyle:'H', triggerThreshold:'L', commentStyle:'H' },
    group: 'attack'
  },
  {
    code: 'GHOST', emoji: '👻', name: '已读不回', nameEn: 'The Ghost Reader',
    tagline: '看到了。（但我不想回。）',
    desc: '你把"已读不回"发展成了一门艺术。不是你不礼貌，而是你觉得有些消息真的不需要回复。"好的""嗯""收到"——这些都是你能给出的最高级别的回应了。你的微信里有47条未回消息，而你正在刷短视频。',
    traits: ['已读不回', '消息黑洞', '选择性失聪', '社交极简'],
    pattern: { commentStyle:'L', shareDesire:'L', socialMask:'L', attentionNeed:'L', digitalDetox:'H' },
    group: 'silent'
  },
  {
    code: 'HAHA', emoji: '🤣', name: '哈哈复读机', nameEn: 'The LOL Repeater',
    tagline: '哈哈哈哈哈哈哈哈哈哈哈哈',
    desc: '你的评论区永远只有一种声音——"哈哈哈哈哈哈"。不管是搞笑视频、沙雕新闻还是朋友的自拍，你的回复永远是"笑死了""哈哈哈哈哈""救命太好笑了"。你不是不会说别的，而是笑已经是你最高效的社交货币了。',
    traits: ['笑死了', '复读机', '快乐源泉', '表情包战神'],
    pattern: { emojiDependency:'H', commentStyle:'M', sarcasmLevel:'L', expressionStyle:'L', attentionNeed:'L' },
    group: 'warm'
  },
  {
    code: 'FISH', emoji: '🐟', name: '摸鱼达人', nameEn: 'The Slacking Pro',
    tagline: '上班的意义就是等下班。',
    desc: '你把上班摸鱼练成了一门绝技。Alt+Tab切屏的速度比任何快捷键都熟练，你能在Excel后面藏一个小说阅读器。你的工作效率并不低——你只是把80%的时间用在了摸鱼上，然后用20%的时间完成了100%的工作。',
    traits: ['摸鱼王者', 'Alt+Tab大师', '效率反差', '带薪冲浪'],
    pattern: { digitalDetox:'L', melonEating:'H', contentCreation:'L', shareDesire:'M', platformLoyalty:'L' },
    group: 'silent'
  },
  {
    code: 'PUSH', emoji: '📢', name: '转发狂魔', nameEn: 'The Share-Everything',
    tagline: '@所有人 快看这个！！！',
    desc: '你的朋友圈和群聊就是一个信息中转站。看到好文章转发，看到好视频转发，看到优惠券转发，看到养生文章也转发。你坚信"好东西要分享"，虽然你分享的东西里有60%别人根本不想看。你的家族群因为你的存在永远不会冷场。',
    traits: ['转发达人', '信息轰炸', '好东西分享', '群聊活跃分子'],
    pattern: { shareDesire:'H', commentStyle:'H', melonEating:'H', attentionNeed:'H', expressionStyle:'M' },
    group: 'spectator'
  },
  {
    code: 'DIVE', emoji: '🤿', name: '深海潜水员', nameEn: 'The Deep Diver',
    tagline: '我在看，但你看不到我。',
    desc: '你是互联网的隐形人。你有微博但从不发，你有朋友圈但永远是空的，你在群里万年潜水从不冒泡。但你什么都知道——谁结婚了、谁离职了、谁又发疯了。你是最安静的观察者，默默地看着这个喧嚣的互联网世界。',
    traits: ['万年潜水', '无痕浏览', '信息收集者', '低存在感'],
    pattern: { commentStyle:'L', shareDesire:'L', expressionStyle:'L', attentionNeed:'L', melonEating:'H' },
    group: 'silent'
  },
  {
    code: 'MASK', emoji: '🎭', name: '面具人', nameEn: 'The Shapeshifter',
    tagline: '网上的我和现实的我？两个人。',
    desc: '你在网上是社交牛逼症，现实中是社恐晚期。你在网上谈笑风生、出口成章，现实中点杯奶茶都要在心里rehearsal三遍。你的网友觉得你是人间宝藏，你的同事觉得你是透明人。这不是精分，这是互联网给你的超能力。',
    traits: ['线上社牛', '线下社恐', '人格分裂', '网络自信'],
    pattern: { onlinePersona:'H', socialMask:'H', expressionStyle:'H', attentionNeed:'M', digitalDetox:'L' },
    group: 'persona'
  },
  {
    code: 'RAGE', emoji: '💢', name: '暴怒哥', nameEn: 'The Rager',
    tagline: '气死我了！！！必须发条动态！！！',
    desc: '你的情绪全写在朋友圈里。被领导骂了发一条，被外卖气到发一条，看到不公正的新闻也要发一条。你的朋友圈就是你的情绪垃圾桶，但奇怪的是——大家都爱看你发疯，因为你说出了所有人想说但不敢说的话。',
    traits: ['情绪外放', '互联网嘴替', '敢怒敢言', '朋友圈发疯'],
    pattern: { triggerThreshold:'L', debateDesire:'H', expressionStyle:'H', opinionStrength:'H', socialMask:'L' },
    group: 'attack'
  },
  {
    code: 'ICE', emoji: '🧊', name: '冷漠路人', nameEn: 'The Cold Bystander',
    tagline: '关我什么事？关你什么事？',
    desc: '你对互联网上的一切都无感。热搜？不看。八卦？无聊。争论？浪费时间。你活在自己的信息茧房里，只关注自己感兴趣的三件事。别人觉得你清高，你觉得自己只是在保护精力。毕竟，这个世界上99%的信息都与你无关。',
    traits: ['万事冷漠', '选择性关注', '精力保护', '信息极简'],
    pattern: { melonEating:'L', debateDesire:'L', shareDesire:'L', triggerThreshold:'H', commentStyle:'L' },
    group: 'silent'
  },
  {
    code: 'ESSAY', emoji: '📝', name: '小作文选手', nameEn: 'The Essay Writer',
    tagline: '长文预警⚠️ 以下是我的2000字深度分析...',
    desc: '你发个朋友圈都要写800字小作文，评论区回复比原帖还长。你不是话多，你只是觉得——不把事情说清楚就对不起自己的表达欲。你的微信消息永远是"对方正在输入中..."长达五分钟，收到的人看到消息长度就想先存个书签。',
    traits: ['长文输出', '深度分析', '表达欲爆棚', '逻辑清晰'],
    pattern: { expressionStyle:'H', contentCreation:'H', opinionStrength:'H', commentStyle:'H', emojiDependency:'L' },
    group: 'creator'
  },
  {
    code: 'KOL', emoji: '🎤', name: '野生意见领袖', nameEn: 'The Grassroot KOL',
    tagline: '听我说，这件事的本质是...',
    desc: '虽然你没有百万粉丝，但在你的小圈子里，你就是意见领袖。朋友买手机要问你，选餐厅要问你，甚至分手都要听你分析。你热爱输出观点，更热爱别人说"确实！"你不知道的是——你的建议有50%是对的，但大家记住的都是对的那50%。',
    traits: ['观点输出', '圈子领袖', '好为人师', '自信满满'],
    pattern: { opinionStrength:'H', expressionStyle:'H', attentionNeed:'H', contentCreation:'H', debateDesire:'M' },
    group: 'creator'
  },
  {
    code: 'SHEEP', emoji: '🐑', name: '跟风羊', nameEn: 'The Trend Follower',
    tagline: '这个好火啊！我也要试试！',
    desc: '互联网上火什么你就玩什么。MBTI火了你测了，SBTI火了你也测了，互联网嘴替测试火了你还是测了（对，就是现在）。你不是没有主见，你只是觉得大家都在玩的东西一定有它好玩的道理。你的手机里有37个最近一周下载的App，其中35个下周就会被卸载。',
    traits: ['追热点', '跟风达人', '尝鲜爱好者', 'App囤积症'],
    pattern: { melonEating:'H', shareDesire:'H', platformLoyalty:'H', onlinePersona:'L', opinionStrength:'L' },
    group: 'spectator'
  },
  {
    code: 'DIG', emoji: '🔍', name: '考古学家', nameEn: 'The Profile Digger',
    tagline: '等等，你2018年发过一条微博...',
    desc: '你的信息搜集能力堪比FBI。给你一个人的微信号，你能在30分钟内翻完他所有社交平台的历史记录。你不是八卦，你只是"好奇心旺盛"。你知道前同事的前任的现任叫什么，你能通过一张照片的背景推断拍摄地点。你是互联网考古界的扛把子。',
    traits: ['信息挖掘', '考古专家', '细节控', 'FBI级搜索'],
    pattern: { melonEating:'H', commentStyle:'L', socialMask:'M', contentCreation:'L', shareDesire:'L' },
    group: 'spectator'
  },
  {
    code: 'EMO', emoji: '😭', name: '网抑云', nameEn: 'The Sad Poster',
    tagline: '到点了，该emo了。',
    desc: '每到深夜你就变成另一个人。白天嘻嘻哈哈，晚上emo到不行。你的网易云音乐评论区是你的树洞，你的深夜朋友圈是你的情绪出口。你不是真的抑郁，你只是在这个高速运转的世界里需要一个喘息的角落。',
    traits: ['深夜emo', '感性动物', '音乐树洞', '间歇性丧'],
    pattern: { triggerThreshold:'L', onlinePersona:'H', socialMask:'H', attentionNeed:'M', expressionStyle:'M' },
    group: 'persona'
  },
  {
    code: 'COPY', emoji: '🤖', name: '复读机', nameEn: 'The Copypasta Bot',
    tagline: '6 6 6 6 6 6 6',
    desc: '你的评论区只有三种内容：复读别人的话、发表情包、以及复读别人的表情包。你不生产内容，你只做内容的搬运工。但这并不妨碍你成为评论区最受欢迎的人——因为在这个时代，一个精准的复读有时比原创更有价值。',
    traits: ['复读高手', '表情包库', '玩梗达人', '评论搬运'],
    pattern: { emojiDependency:'H', contentCreation:'L', expressionStyle:'L', commentStyle:'H', sarcasmLevel:'M' },
    group: 'warm'
  },
  {
    code: 'RAIN', emoji: '🌈', name: '彩虹屁专家', nameEn: 'The Compliment Machine',
    tagline: '天哪！你也太好看了吧！绝绝子！',
    desc: '你是朋友圈的头号捧场王。不管谁发自拍你都能找到夸的角度，不管谁晒娃你都觉得是最可爱的。你的评论永远是"绝了！""太美了！""姐妹你也太会穿了吧！"。你不是在拍马屁，你是真的觉得——每个人都值得被夸奖。',
    traits: ['夸夸达人', '彩虹屁精', '评论区甜心', '真诚赞美'],
    pattern: { commentStyle:'H', expressionStyle:'H', sarcasmLevel:'L', socialMask:'M', attentionNeed:'M' },
    group: 'warm'
  },
  {
    code: 'HOT', emoji: '🔥', name: '热搜体质', nameEn: 'The Hot Topic Magnet',
    tagline: '怎么每次热搜都跟我有关系？',
    desc: '你发的每一条动态都有可能引发一场小型网络事件。你有一种天赋——不管说什么都能精准踩到大家的情绪点。你的朋友圈点赞数永远是最高的，你的评论区永远是最热闹的。有人说你是网红体质，你说你只是比较会说话而已。',
    traits: ['自带流量', '话题制造机', '情绪共鸣', '天生网感'],
    pattern: { attentionNeed:'H', contentCreation:'H', expressionStyle:'H', shareDesire:'H', opinionStrength:'H' },
    group: 'creator'
  },
  {
    code: 'OFF', emoji: '📵', name: '数字隐士', nameEn: 'The Digital Hermit',
    tagline: '手机？能不带就不带。',
    desc: '你是互联网时代的一股清流。当所有人都在刷手机的时候，你在看书；当所有人都在发朋友圈的时候，你在散步。你不是不会用互联网，你只是觉得——现实世界比任何App都有趣。你的朋友经常以为你换了手机号，其实你只是不想回消息。',
    traits: ['数字极简', '现实主义', '手机冷淡', '独立思考'],
    pattern: { digitalDetox:'H', platformLoyalty:'L', attentionNeed:'L', shareDesire:'L', melonEating:'L' },
    group: 'silent'
  },
  {
    code: 'FACT', emoji: '🎯', name: '较真怪', nameEn: 'The Fact Checker',
    tagline: '等等，你这个数据来源是哪里？',
    desc: '你是评论区的事实核查员。有人发了一条"震惊！90%的人不知道的秘密"，你第一反应就是去查证。你纠正过无数谣言，辟过无数谣言，但大家并不领情——因为在互联网上，正确往往比不上有趣。但你不在乎，因为你有一个信念：真相不能被埋没。',
    traits: ['辟谣达人', '数据控', '较真精神', '逻辑在线'],
    pattern: { opinionStrength:'H', debateDesire:'H', sarcasmLevel:'M', contentCreation:'H', triggerThreshold:'M' },
    group: 'attack'
  },
  {
    code: 'PILL', emoji: '💊', name: '毒鸡汤', nameEn: 'The Dark Motivator',
    tagline: '努力不一定成功，但不努力一定很舒服。',
    desc: '你是正能量的反面——你专门生产毒鸡汤。"比你优秀的人比你更努力，所以你努力也没用""条条大路通罗马，但你家就在罗马——的隔壁贫民窟"。你不是真的丧，你只是用一种更真实的方式解构了这个"努力就能成功"的叙事。',
    traits: ['毒鸡汤王', '解构大师', '反鸡汤', '真实残酷'],
    pattern: { sarcasmLevel:'H', expressionStyle:'H', onlinePersona:'M', contentCreation:'H', socialMask:'L' },
    group: 'attack'
  },
  {
    code: 'SHOW', emoji: '🎪', name: '整活达人', nameEn: 'The Content Creator',
    tagline: '等等，让我做个表情包先。',
    desc: '你是互联网的快乐制造机。别人在消费内容，你在创造内容。一个热点出来，你能在10分钟内做出三个表情包和一个短视频。你的创作灵感永远不会枯竭，因为这个荒诞的世界就是你最大的素材库。',
    traits: ['内容创作者', '表情包工厂', '灵感无限', '快乐制造'],
    pattern: { contentCreation:'H', emojiDependency:'H', expressionStyle:'H', shareDesire:'H', attentionNeed:'H' },
    group: 'creator'
  },
  {
    code: 'AUNTIE', emoji: '👵', name: '互联网居委会', nameEn: 'The Internet Auntie',
    tagline: '小伙子/小姑娘，听阿姨一句劝...',
    desc: '你在评论区的角色就是大家的妈/大家的阿姨。看到有人深夜发动态你要劝早睡，看到有人吐槽工作你要劝跳槽，看到有人失恋你要劝分。你的评论永远饱含生活智慧和过来人的经验，虽然有时候别人只是想发个牢骚而已。',
    traits: ['操心命', '过来人', '评论区妈妈', '善意满满'],
    pattern: { commentStyle:'H', debateDesire:'M', socialMask:'L', attentionNeed:'M', sarcasmLevel:'L' },
    group: 'warm'
  }
]

var TYPE_GROUPS = {
  attack:    { name: '输出型选手', desc: '用文字征服世界的人' },
  warm:      { name: '温暖型选手', desc: '互联网的人间温度' },
  spectator: { name: '围观型选手', desc: '看戏不嫌事大的人' },
  silent:    { name: '沉默型选手', desc: '互联网的隐形居民' },
  creator:   { name: '创作型选手', desc: '互联网内容的源头活水' },
  persona:   { name: '人设型选手', desc: '线上线下两幅面孔' }
}

module.exports = {
  DIMENSION_LABELS: DIMENSION_LABELS,
  DIMENSION_KEYS: DIMENSION_KEYS,
  SBTI_TYPES: SBTI_TYPES,
  TYPE_GROUPS: TYPE_GROUPS
}
