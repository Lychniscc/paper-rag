
# 导入os，读取环境变量
import os

# 读取.env文件
from dotenv import load_dotenv

# 使用OpenAI兼容客户端
from openai import OpenAI

# 加载.env配置
load_dotenv()

# 获取第三方API密钥
api_key = os.getenv("LLM_API_KEY")

# 获取第三方接口地址
base_url = os.getenv("LLM_BASE_URL")

# 获取模型ID
model_name = os.getenv("LLM_MODEL")

# 检查配置是否完整
if not all([api_key, base_url, model_name]):
    raise ValueError("请检查.env中的三个配置项")

# 创建API客户端
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=60.0,
    max_retries=0
)

try:
    # 发起一次真实大模型请求
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "请用一句中文解释什么是DOA估计。"
            }
        ],
        max_tokens=150
    )

    # 打印模型回答
    print("模型回答：")
    print(response.choices[0].message.content)

    # 打印模型使用情况
    if response.usage:
        print("输入Token：", response.usage.prompt_tokens)
        print("输出Token：", response.usage.completion_tokens)

except Exception as exc:
    # 只打印错误类别，避免泄露包含认证信息的请求详情
    print("调用失败，错误类型：", type(exc).__name__)
    print("请检查模型ID、分组权限、余额及API端点")
