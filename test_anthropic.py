
# 导入os，用于读取环境变量
import os

# 导入dotenv，用于加载.env文件
from dotenv import load_dotenv

# 导入Anthropic客户端
from anthropic import Anthropic

# 读取.env文件
load_dotenv()

# 获取第三方平台API密钥
api_key = os.getenv("ANTHROPIC_API_KEY")

# 获取API地址
base_url = os.getenv("ANTHROPIC_BASE_URL")

# 获取模型名称
model_name = os.getenv("ANTHROPIC_MODEL")

# 检查配置是否完整
if not all([api_key, base_url, model_name]):
    raise ValueError("请检查.env中的API配置")

# 创建Anthropic客户端
client = Anthropic(
    api_key=api_key,
    base_url=base_url,
    timeout=30.0,
    max_retries=0
)

try:
    # 使用Anthropic Messages API发起请求
    response = client.messages.create(
        model=model_name,
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": "请用一句中文解释什么是DOA估计。"
            }
        ]
    )

    # 输出模型回复
    for block in response.content:
        if block.type == "text":
            print("模型回答：", block.text)

    # 输出本次请求的Token统计
    print("输入Token：", response.usage.input_tokens)
    print("输出Token：", response.usage.output_tokens)

except Exception as exc:
    # 只显示异常类型，避免输出可能包含密钥的信息
    print("调用失败")
    print("错误类型：", type(exc).__name__)

    # 如果服务器返回HTTP错误，输出状态码
    if hasattr(exc, "status_code"):
        print("HTTP状态码：", exc.status_code)

    # 如果存在底层异常，输出其类别
    if exc.__cause__ is not None:
        print(
            "底层错误：",
            type(exc.__cause__).__name__
        )
