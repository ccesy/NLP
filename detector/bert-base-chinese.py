from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载中文情感分类模型
model_name = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

last_digit = 1
second_last_digit = 0

# 影评和外卖评论句子
movie_reviews = [
    "这部电影太精彩了，节奏紧凑毫无冷场，完全沉浸其中！",
    "剧情设定新颖不落俗套，每个转折都让人惊喜。",
    "导演功力深厚，镜头语言非常有张力，每一帧都值得回味。",
    "美术、服装、布景细节丰富，完全是视觉盛宴！",
    "是近年来最值得一看的国产佳作，强烈推荐！",
    "剧情拖沓冗长，中途几次差点睡着。",
    "演员表演浮夸，完全无法让人产生代入感。",
    "剧情老套，充满套路和硬凹的感动。",
    "对白尴尬，像是AI自动生成的剧本。",
    "看完只觉得浪费了两个小时，再也不想看第二遍。"
]

food_reviews = [
    "食物完全凉了，吃起来像隔夜饭，体验极差。",
    "汤汁洒得到处都是，包装太随便了。",
    "味道非常一般，跟评论区说的完全不一样。",
    "分量太少了，照片看着满满的，实际就几口。",
    "食材不新鲜，有异味，感觉不太卫生。",
    "食物份量十足，性价比超高，吃得很满足！",
    "味道超级赞，和店里堂食一样好吃，五星好评！",
    "这家店口味稳定，已经回购好几次了，值得信赖！",
    "点单备注有按要求做，服务意识很棒。",
    "包装环保、整洁美观，整体体验非常好"
]

# 选择对应句子
selected_movie = movie_reviews[last_digit]
selected_food = food_reviews[second_last_digit]

# 情感预测函数
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return "正面" if predicted_class == 1 else "负面"

# 执行预测
movie_sentiment = predict_sentiment(selected_movie)
food_sentiment = predict_sentiment(selected_food)

# 输出
print(f"影评句子：{selected_movie}  情感倾向：{movie_sentiment}")
print(f"外卖评价：{selected_food}  情感倾向：{food_sentiment}")
