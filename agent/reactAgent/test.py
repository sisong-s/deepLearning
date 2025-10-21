from openai import OpenAI # 虽然使用的是阿里云的模型，但阿里云提供了与 OpenAI 兼容的 API 接口，因此可以直接使用 OpenAI 的 SDK
aliyun_api_key = 'sk-9b431d36adce4c7aa022058f02142930'
client = OpenAI(
    api_key=aliyun_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
) # 创建 OpenAI 客户端实例，用于后续调用 API
# 设置 API 的基础 URL 为阿里云 DashScope 服务的兼容模式地址
# 这个 URL 使得 OpenAI SDK 可以与阿里云的 API 服务进行通信
response = client.chat.completions.create( # 调用客户端的聊天补全接口，创建一个对话请求
    model="qwen-max",
    messages=[ # 定义消息列表，用于传递对话历史
        {'role': 'user', 'content': "你是谁？"} # 添加一条用户消息，角色为 "user"，内容为 "你是谁？"
    ]
)

# 打印完整回答内容
print(response.choices[0].message.content)

# response是 API 返回的完整响应对象
# choices是一个列表，包含模型生成的可能回答（通常只有一个）
# message.content是 AI 生成的具体回答文本

# API 接口标准化的兼容设计OpenAI 的 API 有一套标准化的接口格式（包括请求参数、响应结构、认证方式等）。阿里云为了降低用户的迁移成本，在自己的 DashScope 服务中实现了一套与 OpenAI 接口格式完全一致的 "兼容模式"。
# 这意味着你可以直接使用 OpenAI 官方的 Python SDK（openai库），不需要修改代码逻辑，只需更换base_url和api_key，就能调用阿里云的模型（如 qwen-max）。
# 为什么需要这个 URL
# 正常情况下，OpenAI SDK 默认会连接 OpenAI 官方的服务器（api.openai.com）
# 通过修改base_url，我们将请求的目标服务器指向了阿里云的 DashScope 服务
# 这个 URL 中的compatible-mode明确表示启用兼容模式，告诉阿里云服务："请用 OpenAI 的格式来处理我的请求"