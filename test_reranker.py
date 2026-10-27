
# 导入Cross-Encoder模型
from sentence_transformers import CrossEncoder

# 定义Hugging Face模型ID
model_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# 加载模型
# 第一次运行时会自动下载
# 后续运行时优先使用本地缓存
print("正在加载Cross-Encoder模型...")

model = CrossEncoder(
    model_name,
    device="cpu",
    max_length=512
)

print("Cross-Encoder模型加载成功！")

# 设置测试问题
question = "论文采用什么方法实现DOA估计？"

# 设置两个英文候选段落
documents = [
    "The proposed algorithm constructs a PARAFAC data model for DOA estimation.",
    "The RMSE decreases as the SNR increases."
]

# 将问题与候选段落组成输入对
pairs = [
    (question, document)
    for document in documents
]

# 使用Cross-Encoder进行相关性打分
scores = model.predict(pairs)

# 输出分数
for index, score in enumerate(scores):
    print(f"段落{index + 1}的相关性得分：{float(score):.4f}")

print("Reranker测试完成！")
