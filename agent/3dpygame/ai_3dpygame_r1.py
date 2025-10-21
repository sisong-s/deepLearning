# -*- coding: utf-8 -*-
"""
AI 3D PyGame 可视化生成器
使用 DeepSeek R1 推理模型和 OpenAI GPT-4o 生成 PyGame 代码
并通过 browser_use 自动在 Trinket.io 上运行代码

主要功能:
1. 接收用户的 PyGame 查询请求
2. 使用 DeepSeek R1 进行推理和代码生成
3. 使用 OpenAI GPT-4o 提取和优化代码
4. 自动在 Trinket.io 上运行生成的代码

作者: AI Assistant
日期: 2025年
"""

import streamlit as st
from openai import OpenAI
from agno.agent import Agent as AgnoAgent
from agno.models.openai import OpenAIChat as AgnoOpenAIChat
from langchain_openai import ChatOpenAI 
import asyncio
from browser_use import Browser

# 设置 Streamlit 页面配置
st.set_page_config(page_title="PyGame Code Generator", layout="wide")

# 初始化会话状态，用于存储 API 密钥
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "deepseek": "",  # DeepSeek API 密钥
        "openai": ""     # OpenAI API 密钥
    }

# 创建侧边栏用于配置 API 密钥
with st.sidebar:
    st.title("API Keys Configuration")
    
    # DeepSeek API 密钥输入框
    st.session_state.api_keys["deepseek"] = st.text_input(
        "DeepSeek API Key",
        type="password",  # 密码类型，隐藏输入内容
        value=st.session_state.api_keys["deepseek"]
    )
    
    # OpenAI API 密钥输入框
    st.session_state.api_keys["openai"] = st.text_input(
        "OpenAI API Key",
        type="password",  # 密码类型，隐藏输入内容
        value=st.session_state.api_keys["openai"]
    )
    
    # 添加分隔线
    st.markdown("---")
    
    # 显示使用说明
    st.info("""
    📝 使用方法:
    1. 在上方输入你的 API 密钥
    2. 编写你的 PyGame 可视化查询
    3. 点击 '生成代码' 获取代码
    4. 点击 '生成可视化' 来:
       - 打开 Trinket.io PyGame 编辑器
       - 复制粘贴生成的代码
       - 自动运行代码
    """)

# 主界面UI
st.title("🎮 AI 3D Visualizer with DeepSeek R1")

# 示例查询，帮助用户理解如何使用
example_query = "Create a particle system simulation where 100 particles emit from the mouse position and respond to keyboard-controlled wind forces"

# 用户查询输入框
query = st.text_area(
    "输入你的 PyGame 查询:",
    height=70,
    placeholder=f"例如: {example_query}"
)

# 将按钮分为两列布局
col1, col2 = st.columns(2)
generate_code_btn = col1.button("生成代码")  # 生成代码按钮
generate_vis_btn = col2.button("生成可视化")  # 生成可视化按钮

# 处理生成代码按钮点击事件
if generate_code_btn and query:
    # 检查是否提供了必要的 API 密钥
    if not st.session_state.api_keys["deepseek"] or not st.session_state.api_keys["openai"]:
        st.error("请在侧边栏提供 DeepSeek 和 OpenAI 的 API 密钥")
        st.stop()

    # 初始化 DeepSeek 客户端
    deepseek_client = OpenAI(
        api_key=st.session_state.api_keys["deepseek"],
        base_url="https://api.deepseek.com"  # DeepSeek API 基础URL
    )

    # 定义系统提示词，指导 AI 生成高质量的 PyGame 代码
    system_prompt = """你是一个 Pygame 和 Python 专家，专门通过 pygame 和 python 编程制作游戏和可视化。
    在你的推理和思考过程中，请在推理中包含清晰、简洁、格式良好的 Python 代码。
    始终为你提供的代码包含解释说明。"""

    try:
        # 第一步：使用 DeepSeek R1 进行推理
        with st.spinner("正在生成解决方案..."):
            deepseek_response = deepseek_client.chat.completions.create(
                model="deepseek-reasoner",  # 使用 DeepSeek 推理模型
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=1  # 设置最大令牌数
            )

        # 获取 DeepSeek 的推理内容
        reasoning_content = deepseek_response.choices[0].message.reasoning_content
        print("\nDeepseek Reasoning:\n", reasoning_content)  # 调试输出
        
        # 在可展开区域显示 R1 的推理过程
        with st.expander("R1 的推理过程"):      
            st.write(reasoning_content)

        # 第二步：初始化 OpenAI 代理用于代码提取
        openai_agent = AgnoAgent(
            model=AgnoOpenAIChat(
                id="gpt-4o",  # 使用 GPT-4o 模型
                api_key=st.session_state.api_keys["openai"]
            ),
            show_tool_calls=True,  # 显示工具调用
            markdown=True          # 启用 Markdown 格式
        )

        # 定义代码提取提示词
        extraction_prompt = f"""从以下内容中提取纯 Python 代码，这些内容是针对制作 pygame 脚本的特定查询的推理。
        只返回原始代码，不要任何解释或 markdown 反引号:
        {reasoning_content}"""

        # 第三步：提取代码
        with st.spinner("正在提取代码..."):
            code_response = openai_agent.run(extraction_prompt)
            extracted_code = code_response.content

        # 将生成的代码存储在会话状态中，供后续使用
        st.session_state.generated_code = extracted_code
        
        # 在可展开区域显示生成的代码
        with st.expander("生成的 PyGame 代码", expanded=True):      
            st.code(extracted_code, language="python")
            
        st.success("代码生成成功！点击 '生成可视化' 来运行它。")

    except Exception as e:
        # 错误处理
        st.error(f"发生错误: {str(e)}")

# 处理生成可视化按钮点击事件
elif generate_vis_btn:
    # 检查是否已生成代码
    if "generated_code" not in st.session_state:
        st.warning("请先生成代码再进行可视化")
    else:
        # 定义异步函数，在 Trinket.io 上运行 PyGame 代码
        async def run_pygame_on_trinket(code: str) -> None:
            """
            在 Trinket.io 上自动运行 PyGame 代码的异步函数
            
            参数:
                code (str): 要运行的 PyGame 代码
            """
            # 初始化浏览器实例
            browser = Browser()
            from browser_use import Agent 
            
            # 创建浏览器上下文
            async with await browser.new_context() as context:
                # 初始化 ChatOpenAI 模型
                model = ChatOpenAI(
                    model="gpt-4o", 
                    api_key=st.session_state.api_keys["openai"]
                )
                
                # 代理1：导航到 Trinket.io PyGame 页面
                agent1 = Agent(
                    task='前往 https://trinket.io/features/pygame，这是你唯一的任务。',
                    llm=model,
                    browser_context=context,
                )
                
                # 代理2：执行代码（点击运行按钮）
                executor = Agent(
                    task='执行器。通过点击右侧的运行按钮来执行用户编写的代码。',
                    llm=model,
                    browser_context=context
                )

                # 代理3：编码器（等待用户输入代码）
                coder = Agent(
                    task='编码器。你的任务是等待用户在代码编辑器中写入代码，等待10秒。',
                    llm=model,
                    browser_context=context
                )
                
                # 代理4：查看器（观察 PyGame 窗口）
                viewer = Agent(
                    task='查看器。你的任务是观察 pygame 窗口10秒钟。',
                    llm=model,
                    browser_context=context,
                )

                # 显示运行状态并执行代理任务
                with st.spinner("正在 Trinket 上运行代码..."):
                    try:
                        # 按顺序执行各个代理的任务
                        await agent1.run()    # 导航到页面
                        await coder.run()     # 等待代码输入
                        await executor.run()  # 执行代码
                        await viewer.run()    # 观察结果
                        st.success("代码正在 Trinket 上运行！")
                    except Exception as e:
                        # 如果自动运行失败，提供手动操作提示
                        st.error(f"在 Trinket 上运行代码时出错: {str(e)}")
                        st.info("你仍然可以复制上面的代码并在 Trinket 上手动运行")

        # 运行异步函数，使用存储的生成代码
        asyncio.run(run_pygame_on_trinket(st.session_state.generated_code))

# 处理用户点击生成代码但未输入查询的情况
elif generate_code_btn and not query:
    st.warning("请在生成代码前输入查询")