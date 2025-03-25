import re
from jieba import cut
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

def get_words(filename):
    """读取文本并返回分词后的字符串（空格分隔）"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            # 过滤无效字符
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            # 使用jieba分词并过滤长度为1的词
            line = cut(line)
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return ' '.join(words)  # 返回空格分隔的字符串

# 读取训练数据（0.txt~150.txt）
train_files = [f'邮件_files/{i}.txt' for i in range(151)]
corpus = [get_words(f) for f in train_files]

# 计算TF-IDF特征（限制特征数为100）
vectorizer = TfidfVectorizer(max_features=100)
X_train = vectorizer.fit_transform(corpus)

# 准备标签（前127为垃圾邮件，后24为普通邮件）
labels = [1] * 127 + [0] * 24

# 训练模型
model = MultinomialNB()
model.fit(X_train, labels)

def predict(filename):
    """预测新邮件"""
    text = get_words(filename)
    X_new = vectorizer.transform([text])
    result = model.predict(X_new)
    return '垃圾邮件' if result == 1 else '普通邮件'

# 测试
test_files = [f'邮件_files/{i}.txt' for i in range(151, 156)]
for file in test_files:
    print(f'{file} 分类情况: {predict(file)}')