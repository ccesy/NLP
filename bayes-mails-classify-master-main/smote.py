import re
from jieba import cut
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE  # 关键库

def get_words(filename):
    """读取文件并返回分词后的字符串（空格分隔）"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            line = cut(line)
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return ' '.join(words)

# 读取训练数据（0.txt~150.txt）
train_files = [f'邮件_files/{i}.txt' for i in range(151)]
corpus = [get_words(f) for f in train_files]

# 计算TF-IDF特征
vectorizer = TfidfVectorizer(max_features=100)
X_train = vectorizer.fit_transform(corpus)  # 稀疏矩阵格式
labels = [1] * 127 + [0] * 24  # 前127为垃圾邮件，后24为普通邮件

# 使用SMOTE解决样本失衡问题
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, labels)

# 训练模型
model = MultinomialNB()
model.fit(X_resampled, y_resampled)

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