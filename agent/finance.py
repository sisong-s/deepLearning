# 一个简单的金融数据查询Agent示例
from phi.agent import Agent
finance_agent = Agent(name="Finance AI Agent")
finance_agent.print_response("Summarize analyst recommendations for NVDA", stream=True)