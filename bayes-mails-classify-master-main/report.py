import re
import os
import numpy as np
from jieba import cut
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 定义测试集的真实标签（根据实际情况修改）
# 假设测试文件151.txt~155.txt的真实标签如下：
y_test = [1, 0, 1, 0, 1]  # 1:垃圾邮件, 0:普通邮件

def get_words(filename):
    """改进分词函数，处理空文件"""
    words = []
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as fr:
            for line in fr:
                line = line.strip()
                line = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', '', line)
                if not line:
                    continue
                seg_list = cut(line)
                seg_list = [word for word in seg_list if len(word) > 1]
                words.extend(seg_list)
    except Exception as e:
        print(f"文件 {filename} 读取错误: {str(e)}")
    return ' '.join(words) if words else 'unknown_token'

# 加载训练数据
train_files = [f'邮件_files/{i}.txt' for i in range(151) if os.path.exists(f'邮件_files/{i}.txt')]
corpus_train = [get_words(f) for f in train_files]
labels_train = [1] * 127 + [0] * 24  # 前127为垃圾邮件，后24为普通邮件

# 特征提取
vectorizer = TfidfVectorizer(max_features=100)
X_train = vectorizer.fit_transform(corpus_train)

# 处理样本失衡
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, labels_train)

# 训练模型
model = MultinomialNB()
model.fit(X_res, y_res)

# 加载测试数据
test_files = [f'邮件_files/{i}.txt' for i in range(151, 156) if os.path.exists(f'邮件_files/{i}.txt')]
corpus_test = [get_words(f) for f in test_files]
X_test = vectorizer.transform(corpus_test)

# 预测测试集
y_pred = model.predict(X_test)

# 输出分类报告
print("\n=== 分类评估报告 ===")
print(classification_report(y_test, y_pred, target_names=["普通邮件", "垃圾邮件"]))

# 保留原有预测函数
def predict(filename):
    """预测单个文件"""
    text = get_words(filename)
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    return {
        'prediction': '垃圾邮件' if pred == 1 else '普通邮件',
        'spam_prob': f"{proba[1]:.2%}",
        'normal_prob': f"{proba[0]:.2%}"
    }

# 示例预测
print("\n=== 单文件预测示例 ===")
for file in test_files:
    res = predict(file)
    print(f"{os.path.basename(file)}: {res['prediction']} (垃圾邮件概率: {res['spam_prob']})")