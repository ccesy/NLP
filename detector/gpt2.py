from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

# 选择模型
model_name = "uer/gpt2-chinese-cluecorpussmall"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)

# 设置学号末尾索引（例如：1）
last_digit = 1

# 句子开头选项
prompts = [
    "如果我拥有一台时间机器",
    "当人类第一次踏上火星",
    "如果动物会说话，它们最想告诉人类的是",
    "有一天，城市突然停电了",
    "当我醒来，发现自己变成了一本书",
    "假如我能隐身一天，我会",
    "我走进了那扇从未打开过的门",
    "在一个没有网络的世界里",
    "如果世界上只剩下我一个人",
    "梦中醒来，一切都变了模样"
]

# 选择开头
prompt = prompts[last_digit]

# 编码输入
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# 生成文本
output = model.generate(
    input_ids,
    max_length=1000,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
    num_return_sequences=1
)

# 解码输出
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 打印结果
print("生成结果：")
print(generated_text)
