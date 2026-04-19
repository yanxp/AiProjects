/**
 * SBTI 27 Personality Types
 * 5 Models × 3 Dimensions = 15 Dimensions
 *
 * Models:
 *   Self (自我):       selfEsteem, selfClarity, coreValue
 *   Emotion (情感):    attachSecurity, emotionalInvest, boundaries
 *   Attitude (态度):   worldview, ruleFlexibility, meaningfulness
 *   Action (行动):     motivation, decisionStyle, executionMode
 *   Social (社交):     socialInitiative, interpersonalBound, authenticity
 *
 * Each dimension score: L (low), M (mid), H (high)
 */

const DIMENSION_LABELS = {
  selfEsteem:         '自尊水平',
  selfClarity:        '自我清晰度',
  coreValue:          '核心价值感',
  attachSecurity:     '依恋安全感',
  emotionalInvest:    '情感投入度',
  boundaries:         '情感边界',
  worldview:          '世界观',
  ruleFlexibility:    '规则弹性',
  meaningfulness:     '意义感',
  motivation:         '内驱力',
  decisionStyle:      '决策风格',
  executionMode:      '执行模式',
  socialInitiative:   '社交主动性',
  interpersonalBound: '人际边界',
  authenticity:       '真实性'
};

const DIMENSION_KEYS = Object.keys(DIMENSION_LABELS);

const SBTI_TYPES = [
  {
    code: 'CTRL',
    emoji: '🎮',
    name: '拿捏者',
    nameEn: 'The Puppeteer',
    tagline: '怎么样，是不是被我拿捏了？',
    desc: '你就是那个永远掌控全局的人。情绪稳定得像个机器人，社交游刃有余，别人还在崩溃的时候你已经在制定Plan B了。在一个充满自嘲的测试里，你居然拿到了最强结果——说明你要么真的很强，要么自我认知有点过于良好。',
    traits: ['掌控全局', '情绪稳定', '社交达人', '计划周密'],
    pattern: { selfEsteem:'H', selfClarity:'H', coreValue:'H', motivation:'H', decisionStyle:'H', executionMode:'H', socialInitiative:'H', authenticity:'H' },
    group: 'clear'
  },
  {
    code: 'BOSS',
    emoji: '👑',
    name: '领导者',
    nameEn: 'The Boss',
    tagline: '方向盘给我，我来开。',
    desc: '你天生就有一种"给我上"的气场。不管是组织聚会还是分配工作，你总是自然而然地站在C位。但你有没有想过，也许不是所有人都想被你领导？不过没关系，反正他们最后都会听你的。',
    traits: ['天生领袖', '行动力强', '气场十足', '目标明确'],
    pattern: { selfEsteem:'H', coreValue:'H', motivation:'H', executionMode:'H', socialInitiative:'H' },
    group: 'clear'
  },
  {
    code: 'GOGO',
    emoji: '🏃',
    name: '行者',
    nameEn: 'The Go-Getter',
    tagline: 'Gogogo~ 冲就完事了！',
    desc: '你是行动派中的战斗机。想到就做，做了再说，错了再改。当别人还在纠结"要不要做"的时候，你已经做完了第三版。你的字典里没有"拖延"两个字，但偶尔也会因为冲太快而翻车。',
    traits: ['雷厉风行', '执行力爆表', '不纠结', '偶尔翻车'],
    pattern: { motivation:'H', executionMode:'H', decisionStyle:'H', ruleFlexibility:'H' },
    group: 'clear'
  },
  {
    code: 'THIN-K',
    emoji: '🧠',
    name: '思考者',
    nameEn: 'The Thinker',
    tagline: '深度思考中...已过100秒。',
    desc: '你的大脑就像一台永不停歇的分析机器。每件事都要想三遍，每个决定都要考虑十七种可能性。朋友们觉得你"想太多"，但你觉得他们"想太少"。唯一的问题是：等你想清楚了，机会可能已经过去了。',
    traits: ['深度思考', '理性分析', '犹豫不决', '洞察力强'],
    pattern: { selfClarity:'H', worldview:'H', meaningfulness:'H', decisionStyle:'L' },
    group: 'clear'
  },
  {
    code: 'MONK',
    emoji: '🧘',
    name: '僧人',
    nameEn: 'The Monk',
    tagline: '万事皆空，不如躺平。',
    desc: '你已经看透了这个世界的本质——一切都是虚无。别人在卷的时候你在冥想，别人在焦虑的时候你在喝茶。你不是不在乎，你是已经超越了"在乎"这个层面。佛系到极致就是你。',
    traits: ['看透世事', '内心平静', '佛系人生', '独善其身'],
    pattern: { selfClarity:'H', boundaries:'H', worldview:'H', meaningfulness:'L', socialInitiative:'L', interpersonalBound:'H' },
    group: 'clear'
  },
  {
    code: 'DEAD',
    emoji: '💀',
    name: '已死之人',
    nameEn: 'The Walking Void',
    tagline: '我...还活着吗？',
    desc: '恭喜你，你在人格测试中测出了"已死"。每天起床就是一场修行，上班像是在渡劫。你的能量条永远显示0%，但神奇的是你还在正常运转。也许这就是现代人的超能力——即使社会性死亡了也能继续搬砖。',
    traits: ['社会性死亡', '能量归零', '麻木前行', '灵魂出窍'],
    pattern: { selfEsteem:'L', coreValue:'L', motivation:'L', meaningfulness:'L', emotionalInvest:'L' },
    group: 'dark'
  },
  {
    code: 'SHIT',
    emoji: '💩',
    name: '喷子',
    nameEn: 'The Cynic',
    tagline: '这个世界就是一坨屎。',
    desc: '你是那种嘴上说着"都是垃圾"但实际上比谁都认真的人。毒舌是你的保护色，吐槽是你的生存技能。你不是真的觉得一切都很糟糕，你只是对这个世界有着更高的标准。某种意义上，喷子才是最后的理想主义者。',
    traits: ['毒舌达人', '暗黑理想主义', '嘴硬心软', '标准极高'],
    pattern: { worldview:'L', ruleFlexibility:'L', authenticity:'H', boundaries:'H' },
    group: 'dark'
  },
  {
    code: 'SOLO',
    emoji: '🌙',
    name: '孤儿',
    nameEn: 'The Lone Wolf',
    tagline: '我哭了...怎么我是孤儿？',
    desc: '你就像一只刺猬，用满身的刺来保护那颗柔软的心。你不是不想社交，而是每次社交都感觉在消耗生命值。一个人待着的时候你最放松，但偶尔深夜也会突然觉得有点孤单。别担心，你不是真的孤儿——你只是享受孤独罢了。',
    traits: ['独来独往', '内心敏感', '保护壳厚', '享受独处'],
    pattern: { attachSecurity:'L', socialInitiative:'L', interpersonalBound:'H', selfEsteem:'L' },
    group: 'dark'
  },
  {
    code: 'IMSB',
    emoji: '🪩',
    name: '自毁者',
    nameEn: 'Self-Destruct Mode',
    tagline: '等等...我真的是傻X吗？',
    desc: '你的内心戏比电视剧还精彩。一件小事就能在脑子里翻拍成8集连续剧，而且每一集的结局都是"我果然不行"。你是自我PUA的王者，明明很优秀却永远觉得自己不够好。停下来想想——也许你唯一的问题就是对自己太苛刻了。',
    traits: ['内心戏王', '自我怀疑', '完美主义', '过度反思'],
    pattern: { selfEsteem:'L', selfClarity:'L', emotionalInvest:'H', boundaries:'L' },
    group: 'dark'
  },
  {
    code: 'ZZZZ',
    emoji: '😴',
    name: '装死达人',
    nameEn: 'The Playing-Dead Pro',
    tagline: '我没死，我只是在睡觉。',
    desc: '你是拖延症晚期患者中的佼佼者。deadline之前的你和deadline那天的你完全是两个人——前者是一条咸鱼，后者是一台永动机。你深谙"在最后一秒爆发"的艺术，问题是每次都心脏病发作般地完成任务。',
    traits: ['拖延晚期', 'Deadline战士', '平时装死', '关键时刻爆发'],
    pattern: { motivation:'L', executionMode:'L', ruleFlexibility:'H', meaningfulness:'M' },
    group: 'dark'
  },
  {
    code: 'ATM-er',
    emoji: '💸',
    name: '提款机',
    nameEn: 'The Human ATM',
    tagline: '你以为我很有钱吗？',
    desc: '你是朋友圈里公认的"好人"。永远在付账、永远在帮忙、永远把自己的需求排在最后面。你的爱的语言就是"给予"，给到最后连自己都快被掏空了。也许是时候学会说"不"了——你又不是真的ATM，你也需要充值的。',
    traits: ['无私奉献', '有求必应', '烂好人', '需要学会拒绝'],
    pattern: { emotionalInvest:'H', boundaries:'L', socialInitiative:'H', interpersonalBound:'L', attachSecurity:'H' },
    group: 'grind'
  },
  {
    code: 'MALO',
    emoji: '🐒',
    name: '猴子',
    nameEn: 'The Monkey Brain',
    tagline: '人生就是一场副本，我只是一只小猴子。',
    desc: '你的人生就像是在打游戏——每天都是新副本，每个任务都随机触发。你有着用不完的好奇心和停不下来的精力，一会儿想学这个一会儿想做那个。你不是三分钟热度，你只是对这个世界充满了热情（虽然持续时间确实有点短）。',
    traits: ['好奇心爆棚', '活力四射', '三分钟热度', '快乐至上'],
    pattern: { ruleFlexibility:'H', motivation:'M', executionMode:'M', socialInitiative:'H', authenticity:'H' },
    group: 'grind'
  },
  {
    code: 'FUCK',
    emoji: '🔥',
    name: '野草',
    nameEn: 'The Born Rebel',
    tagline: '野火烧不尽，春风吹又生。',
    desc: '你是打不死的小强，是烧不尽的野草。生活虐你千百遍，你依然活蹦乱跳。你不是不会受伤，而是受伤之后恢复得特别快。这种"去他的，老子继续"的精神，是你最大的超能力。',
    traits: ['打不死', '恢复力强', '叛逆精神', '生命力旺盛'],
    pattern: { selfEsteem:'H', coreValue:'H', worldview:'L', ruleFlexibility:'H', motivation:'H' },
    group: 'grind'
  },
  {
    code: 'JOKE-R',
    emoji: '🤡',
    name: '小丑',
    nameEn: 'The Clown',
    tagline: '原来我们都是小丑。',
    desc: '你是人群中的开心果，是聚会上的气氛担当。你用幽默来化解尴尬，用笑话来掩饰脆弱。只有深夜卸下面具的时候，你才会偷偷想："为什么我总是在逗别人笑，却没人在意我笑不笑？"',
    traits: ['气氛担当', '幽默大师', '外向内伤', '用笑容伪装'],
    pattern: { socialInitiative:'H', authenticity:'L', emotionalInvest:'M', selfEsteem:'M', boundaries:'L' },
    group: 'grind'
  },
  {
    code: 'WOC!',
    emoji: '😱',
    name: '卧槽人',
    nameEn: 'The WTF Person',
    tagline: '卧槽，怎么我是这个人格？！',
    desc: '你对这个世界永远充满惊讶。"卧槽这也行？""卧槽那也可以？"是你的口头禅。你不是大惊小怪，你只是对生活中的每一件事都保持着新鲜感。某种意义上，你是最会发现生活乐趣的人。',
    traits: ['大惊小怪', '好奇宝宝', '表情丰富', '情绪外放'],
    pattern: { emotionalInvest:'H', ruleFlexibility:'M', socialInitiative:'M', authenticity:'H', worldview:'M' },
    group: 'grind'
  },
  {
    code: 'LOVE-R',
    emoji: '💔',
    name: '恋人',
    nameEn: 'The Hopeless Romantic',
    tagline: '爱太满了，现实装不下。',
    desc: '你是为爱而生的人。每一段感情你都全力以赴，每一次心动你都当成最后一次。你的情感浓度比浓缩咖啡还浓，甜蜜到齁。唯一的问题是——你给出的爱太多了，多到有时候连自己都养不活。',
    traits: ['恋爱脑', '全力以赴', '情感丰富', '容易受伤'],
    pattern: { emotionalInvest:'H', attachSecurity:'M', boundaries:'L', selfEsteem:'M', authenticity:'H' },
    group: 'sensitive'
  },
  {
    code: 'MUM',
    emoji: '🤱',
    name: '妈妈',
    nameEn: 'The Mom Friend',
    tagline: '要不...我能叫你一声妈吗？',
    desc: '你操心的范围已经超出了正常人的边界。朋友没吃饭你担心，同事加班你心疼，路边的流浪猫你也要管。你是全世界的妈妈，唯独忘了好好照顾自己。记住：先给自己戴上氧气面罩，再去帮助别人。',
    traits: ['超级共情', '操心命', '温暖治愈', '忘了自己'],
    pattern: { emotionalInvest:'H', boundaries:'L', socialInitiative:'H', attachSecurity:'H', interpersonalBound:'L' },
    group: 'sensitive'
  },
  {
    code: 'FAKE',
    emoji: '🎭',
    name: '假人',
    nameEn: 'The Shapeshifter',
    tagline: '面具戴太久了，忘了真实的脸。',
    desc: '你是社交变色龙，在不同的人面前展现不同的自己。不是因为你虚伪，而是因为你太擅长察言观色了。你能在三秒内读懂一个人需要什么，然后变成他们想要的样子。但夜深人静的时候，你会不会想——真正的"我"到底是谁？',
    traits: ['社交变色龙', '察言观色', '面具高手', '迷失自我'],
    pattern: { authenticity:'L', socialInitiative:'H', selfClarity:'L', interpersonalBound:'L' },
    group: 'sensitive'
  },
  {
    code: 'SEXY',
    emoji: '💃',
    name: '万人迷',
    nameEn: 'The Stunner',
    tagline: '你天生就是焦点！',
    desc: '你走到哪里都自带聚光灯。不是你故意吸引注意力，而是你的气场就是这么强。你可能自己都没意识到——你笑的时候，周围的人都在偷偷看你。这种天生的魅力是上天给你的礼物，好好用它吧。',
    traits: ['天生魅力', '自带聚光灯', '气场强大', '无意识吸引'],
    pattern: { selfEsteem:'H', socialInitiative:'H', authenticity:'H', coreValue:'H', emotionalInvest:'M' },
    group: 'sensitive'
  },
  {
    code: 'OH-NO',
    emoji: '😰',
    name: '完了人',
    nameEn: 'The Oh-No Person',
    tagline: '完了！怎么我是这个人格？！',
    desc: '你的内心有一个永远在拉警报的小人。"完了，邮件发错人了""完了，他是不是生气了""完了，这个DDL要赶不上了"。你不是真的觉得天要塌了，你只是比别人多了一根敏感的神经。好消息是——90%你担心的事情都不会发生。',
    traits: ['焦虑体质', '风险预警', '细节敏感', '杞人忧天'],
    pattern: { attachSecurity:'L', worldview:'L', meaningfulness:'M', selfEsteem:'M', emotionalInvest:'H' },
    group: 'sensitive'
  },
  {
    code: 'Dior-s',
    emoji: '🧢',
    name: '屌丝',
    nameEn: 'The Underdog',
    tagline: '你等着，我的逆袭剧本还没开始呢。',
    desc: '你是那个永远觉得"我还不行"但实际上挺行的人。你自嘲是屌丝，但你的努力程度比很多"人生赢家"都要强。你在等一个逆袭的机会——也许那个机会早就来了，只是你还没注意到而已。',
    traits: ['自嘲达人', '暗中努力', '等待逆袭', '低调实力'],
    pattern: { selfEsteem:'L', coreValue:'M', motivation:'H', meaningfulness:'H', authenticity:'M' },
    group: 'grind'
  },
  {
    code: 'THAN-K',
    emoji: '🙏',
    name: '感恩人',
    nameEn: 'The Gratitude Guru',
    tagline: '感谢天！感谢地！感谢命运让我们相遇！',
    desc: '你是正能量的化身，每天都在发现值得感恩的事情。别人觉得你太"正"了，但你觉得这个世界本来就值得感恩。你的存在就像一盏小灯——虽然不刺眼，但在黑暗中格外温暖。',
    traits: ['正能量满满', '知足常乐', '温暖他人', '感恩一切'],
    pattern: { worldview:'H', meaningfulness:'H', emotionalInvest:'H', attachSecurity:'H', authenticity:'H' },
    group: 'clear'
  },
  {
    code: 'OJBK',
    emoji: '😎',
    name: '随意人',
    nameEn: 'The IDGAF',
    tagline: '我说随便的时候，我是真的随便。',
    desc: '你是"佛系"的另一种表达。吃什么随便，去哪里随便，人生选择也随便。但你的"随便"不是敷衍，而是一种对生活的豁达。你已经参透了一个道理——纠结的时间成本远大于随便选一个的代价。',
    traits: ['真正佛系', '不纠结', '随遇而安', '内心平和'],
    pattern: { ruleFlexibility:'H', decisionStyle:'H', interpersonalBound:'H', meaningfulness:'M', boundaries:'H' },
    group: 'clear'
  },
  {
    code: 'POOR',
    emoji: '💰',
    name: '穷人',
    nameEn: 'The Broke Specialist',
    tagline: '我是穷，但我穷得很专注。',
    desc: '你可能银行卡余额不多，但你的精神世界非常富有。你把有限的资源投入到你真正热爱的事情上——可能是一个冷门爱好，可能是一个不赚钱的梦想。你不是不会赚钱，你只是觉得有些东西比钱重要。',
    traits: ['精神富有', '专注热爱', '资源有限', '梦想坚守'],
    pattern: { coreValue:'H', motivation:'H', meaningfulness:'H', socialInitiative:'L', selfEsteem:'M' },
    group: 'grind'
  },
  {
    code: 'GHOST',
    emoji: '👻',
    name: '鬼魂',
    nameEn: 'The Ghost',
    tagline: '我在，但也不完全在。',
    desc: '你是隐身状态的王者。群聊里你永远潜水，聚会上你总是默默坐在角落。不是你不想参与，而是你更喜欢观察。你就像一个安静的旁观者，看着这个喧嚣的世界，偶尔露个脸证明自己还活着。',
    traits: ['隐身高手', '安静观察', '低存在感', '潜水专家'],
    pattern: { socialInitiative:'L', interpersonalBound:'H', authenticity:'M', selfClarity:'M', executionMode:'L' },
    group: 'dark'
  },
  {
    code: 'HHHH',
    emoji: '🤣',
    name: '哈哈怪',
    nameEn: 'The Cackler',
    tagline: '哈哈哈哈哈哈哈哈（停不下来）',
    desc: '你笑点极低，任何事情都能让你笑出声。朋友圈里的快乐源泉就是你。你的笑声有魔力——不管多沉闷的场合，你一笑大家都忍不住跟着笑。你用笑声对抗生活的苦，虽然有时候笑着笑着就哭了。',
    traits: ['笑点极低', '快乐源泉', '感染力强', '乐观面对'],
    pattern: { emotionalInvest:'M', socialInitiative:'H', authenticity:'H', worldview:'H', ruleFlexibility:'H', motivation:'M' },
    group: 'sensitive'
  },
  {
    code: 'DRUNK',
    emoji: '🍺',
    name: '酒鬼',
    nameEn: 'The Drunkard',
    tagline: '人生得意须尽欢，来，干杯！',
    desc: '你是酒桌上的灵魂人物（虽然这个测试本来就是为了劝人戒酒才做的）。你活在当下，及时行乐，觉得人生苦短不如喝一杯。你的社交方式就是"来，走一个"，解决问题的方式也是"来，走一个"。也许是时候想想别的解压方式了？',
    traits: ['及时行乐', '社交达人', '活在当下', '需要戒酒'],
    pattern: { ruleFlexibility:'H', socialInitiative:'H', boundaries:'L', executionMode:'L', meaningfulness:'L', worldview:'L' },
    group: 'dark'
  }
];

// Group metadata
const TYPE_GROUPS = {
  clear:     { name: '人间清醒组', desc: '看透世界的清醒者们' },
  dark:      { name: '暗黑虚无组', desc: '在黑暗中前行的勇者们' },
  grind:     { name: '搞事混沌组', desc: '在混乱中创造奇迹的人' },
  sensitive: { name: '敏感创造组', desc: '用细腻感知世界的人' }
};
