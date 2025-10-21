import re 
import requests
from openai import OpenAI
aliyun_api_key = 'sk-9b431d36adce4c7aa022058f02142930'
client = OpenAI(
    api_key = aliyun_api_key,
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
class Agent:
    def __init__(self, system = ''):
        self.system = system # 保存系统提示词，也就是我们设定的大模型身份
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({'role':'user', 'content':message})   
        result = self.execute()
        self.messages.append({'role':'assistant', 'content':result}) # 最后保存大模型回复的信息也会以格式化的形式添加到self.messages 中
        return result

    def execute(self):
        response = client.chat.completions.create(
            model = 'qwen-max',
            messages = self.messages
        )
        return response.choices[0].message.content

# abot = Agent('你是一个乐于助人的机器人')
# result = abot('你是谁')
# print(result)
# result = abot('上一个问题的问答分别是什么')
# print(result)

# 错误信息"null is not one of ['system', 'assistant', 'user', 'tool', 'function'] - 'messages.['0].role'"说明：
# 你发送的第一条消息（索引为 0 的消息）的role值为null（空值）
# 而 API 要求role必须是预定义的这几个值之一：system、assistant、user、tool或function

def calculate(what):
    return eval(what)
print(calculate("3 + 7 * 2"))
print(calculate("10 / 4"))

def average_dog_weight(name):
    if name in "Scottish Terrier": 
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")

# 新增：调用外部API服务的函数
def get_external_data(resource, params=None):
    """
    调用外部API服务（JSONPlaceholder）获取数据
    JSONPlaceholder是一个免费的在线REST API，用于测试和原型开发
    """
    base_url = "https://jsonplaceholder.typicode.com"
    
    try:
        # 构建请求URL
        url = f"{base_url}/{resource}"
        
        # 发送GET请求到外部服务
        response = requests.get(url, params=params)
        response.raise_for_status()  # 如果请求失败，抛出异常
        
        # 解析JSON响应
        data = response.json()
        
        # 简化返回结果，只取前3条数据（避免信息过多）
        if isinstance(data, list) and len(data) > 3:
            return f"获取到{resource}数据（前3条）: {str(data[:3])[:200]}..."
        else:
            return f"获取到{resource}数据: {str(data)[:200]}..."
            
    except Exception as e:
        return f"调用外部服务失败: {str(e)}"

average_dog_weight("Scottish Terrier")  
# 返回 "Scottish Terriers average 20 lbs"
average_dog_weight("Labrador")          
# 返回 "An average dog weights 50 lbs"

known_actions = {
    'calculate': calculate,
    'average_dog_weight': average_dog_weight,
    'get_external_data': get_external_data  # 新增外部服务调用工具
}
# 第二部分是解释各个关键结构，Thought 表示模型的思考过程，Action 用于调用工具，
# PAUSE 代表暂时停止以等待外部执行，Observation 是动作执行后的反馈信息。
# 这一段为模型提供了清晰的格式指南，确保它知道每一步该怎么做。

# 第三部分列出了当前模型可调用的两个工具，
# 一个是 calculate 用于执行数学计算，
# 一个是 average_dog_weight 用于查询狗的平均体重。每个工具都有调用示例和用途描述，
# 明确告诉模型能干什么、怎么写，帮助它在需要时做出正确选择。

# 这个提示词是一个典型的 ReAct 模式 Prompt，用于引导大语言模型按 
# “Thought → Action → Observation → Answer” 的流程进行多步推理。
# ！！！！ 当上一个observe结束后，只有模型认为可以给出答案的时候输出answer，
# 否则继续thought，一直循环下去。
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

get_external_data:
e.g. get_external_data: posts
e.g. get_external_data: users?username=Bret
Calls an external service to get data. The first parameter is the resource (posts, users, comments, etc.)
You can add query parameters with ?key=value

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip() # 移除字符串首尾的空白字符（包括空格、换行符 \n、制表符 \t 等）。

# ^Action: 表示必须以 Action: 开头
# (\w+) 匹配一个由字母、数字或下划线组成的字符串，用来捕获工具名
# : 是第二个冒号
# (.*) 匹配这一行剩下的所有内容，作为工具的输入参数
# $ 表示这一行的末尾
action_re = re.compile('^Action: (\w+): (.*)$')
def query(question, max_turns = 5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a)
            for a in result.split('\n')
            if action_re.match(a)
        ]
        if actions:
            # 注意这里只取第一个 Action，意味着如果模型一次输出多个动作，
            # 只会执行第一个。你可以扩展为多动作支持，但这个例子中是单轮单动作执行。
            # groups() 提取正则表达式中括号捕获的具体内容
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception('Unkown action: {}:{}'.format(action, action_input))
            observation = known_actions[action](action_input)
            print('observation:', observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return result

# question = """I have 2 dogs, a border collie and a scottish terrier. \
# What is their combined weight"""
# query(question)
# 测试新增的外部服务调用
print("\n===== 测试外部服务调用 =====")
external_question = "从外部服务获取一些用户数据，然后告诉我第一个用户的名字和邮箱是什么？"
query(external_question)

print("\n===== 测试复杂外部服务查询 =====")
complex_question = "获取外部服务中的帖子数据，然后计算一下返回了多少条帖子？"
query(complex_question)