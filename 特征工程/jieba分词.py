"""
https://github.com/fxsjy/jieba
"""
#引入jieba模块
#集成百度LAC
import jieba


if __name__  == '__main__':
    # 初始化分词模型 词库匹配再进行分词 历史语料库 可以不初始化，默认在第一次运行的时候导入
    # 单例模式 只存在一个实例化对象
    jieba.initialize()
    # jieba.enable_paddle()#开启百度LAC
    words_a = '上海自来水来自海上，所以吃葡萄不吐葡萄皮'
    seg_a = jieba.cut(words_a, cut_all=True)
    print("全模式：", "/".join(seg_a))
    seg_b = jieba.cut(words_a)  # 节省内存空间
    # print(list(seg_b))
    seg_b = jieba.lcut(words_a)  # 方便复用
    print(seg_b)
    print("全模式：", "/".join(seg_b))
    seg_c = jieba.cut_for_search(words_a)
    print("搜索引擎模式", "/".join(seg_c))
    # print(seg_b)
    # <generator object Tokenizer.cut at 0x000002211770C0B0> 分词结果是一个生成器
    print(list(seg_b))  # [] 生成器只能取一遍

    # 添加和删除自定义词汇
    words_a1 = '我为机器学习疯狂打call'
    print("自定义前：", ",".join(jieba.cut(words_a1)))
    jieba.add_word("打call")  # 添加单词，在后续的分词中，遇到分词的时候会认为属于同一个单词
    print("加入'打call'后：", "/".join(jieba.cut(words_a1)))
    jieba.del_word("打call")  # 删除单词，在后续的分词的时候，被删除的不会被认为是一个单词
    print("删除'打call'后：", "/".join(jieba.cut(words_a1)))
    jieba.add_word("机器学习")
    print("加入'机器学习'后：", "/".join(jieba.cut(words_a1)))
    jieba.add_word("《打call》")  # 这里add无效

    # 导入自定义词典
    jieba.del_word("打call")  # 删除之前添加的词汇
    words_a2 = '在正义者联盟的电影里，嘻哈侠和蝙蝠侠联手打败了大boss,我高喊666，为他们疯狂打call'
    print("加载自定义库前：", ",".join(jieba.cut(words_a2)))
    jieba.load_userdict("./datas/01mydict.txt")
    print("加载自定义库后：", ",".join(jieba.cut(words_a2)))
    # 获取切词后的数据
    lsl = []
    for item in jieba.cut(words_a2):
        lsl.append(item)
    print(lsl)
    # 用lcut直接获取切词后的list列表数据
    ls2 = jieba.lcut(words_a2)
    print(ls2)

    # 调整词典，关闭HMM发现新词功能（主要在开发过程中使用）
    print(",".join(jieba.cut('中将如果将徽章放在post中将出错？', HMM=False)))
    print(",".join(jieba.cut('中将如果将徽章放在post中将出错？', HMM=True)))  # 有发现新词的可能性
    jieba.suggest_freq(('中', '将'), True)
    print("=================")
    print(",".join(jieba.cut('如果放在post中将出错。', HMM=False)))
    print(",".join(jieba.cut('如果放在post中将出错。', HMM=True)))  # 有发现新词的可能性
    jieba.suggest_freq(('台', '中'), True)
    print(",".join(jieba.cut('【台中】正确应该不会被切开', HMM=False)))
    jieba.suggest_freq('台中', True)
    print(",".join(jieba.cut('【台中】正确应该不会被切开。', HMM=False)))

    # 关键词提取 对数据降维  特征提取
    # 获取IF-IDF最大的五个单词

    import jieba.analyse

    words_a2 = '在正义者联盟的电影里，嘻哈侠和蝙蝠侠联手打败了大boss,我高喊666，为他们疯狂打call'
    print(jieba.analyse.extract_tags(words_a2, topK=5, withWeight=True)
          )
