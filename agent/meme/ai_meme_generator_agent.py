# 导入必要的库
import asyncio  # 异步编程支持
import streamlit as st  # Streamlit Web应用框架
from browser_use import Agent, SystemPrompt  # 浏览器自动化代理
from langchain_openai import ChatOpenAI  # OpenAI和Deepseek模型接口
from langchain_anthropic import ChatAnthropic  # Claude模型接口
from langchain_core.messages import HumanMessage  # 消息处理
import re  # 正则表达式库，用于URL提取

# ------------------- 新增：解决 asyncio 子进程问题 -------------------
import sys
# 仅在 Windows 系统下设置 ProactorEventLoop（支持创建子进程）
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


async def generate_meme(query: str, model_choice: str, api_key: str) -> str:
    """
    异步生成表情包的核心函数
    
    参数:
        query (str): 用户输入的表情包创意描述
        model_choice (str): 选择的AI模型 ("Claude", "Deepseek", "OpenAI")
        api_key (str): 对应模型的API密钥
    
    返回:
        str: 生成的表情包图片URL，失败时返回None
    """
    # 根据用户选择初始化相应的大语言模型
    if model_choice == "Claude":
        # 初始化Claude模型 - Anthropic的AI助手
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",  # 使用最新的Claude 3.5 Sonnet模型
            api_key=api_key
        )
    elif model_choice == "Deepseek":
        # 初始化Deepseek模型 - 中国的开源AI模型
        llm = ChatOpenAI(
            base_url='https://api.deepseek.com/v1',  # Deepseek API端点
            model='deepseek-chat',  # 使用deepseek-chat模型
            api_key=api_key,
            temperature=0.3  # 设置较低的温度值，保持输出相对稳定
        )
    else:  # OpenAI
        # 初始化OpenAI GPT-4o模型
        llm = ChatOpenAI(
            model="gpt-4o",  # 使用GPT-4o模型，支持视觉功能
            api_key=api_key,
            temperature=0.0  # 设置温度为0，确保输出的确定性
        )

    # 构建详细的任务描述，指导AI代理如何生成表情包
    task_description = (
        "You are a meme generator expert. You are given a query and you need to generate a meme for it.\n"
        "1. Go to https://imgflip.com/memetemplates \n"  # 访问ImgFlip表情包模板网站
        "2. Click on the Search bar in the middle and search for ONLY ONE MAIN ACTION VERB (like 'bully', 'laugh', 'cry') in this query: '{0}'\n"  # 搜索关键动作词
        "3. Choose any meme template that metaphorically fits the meme topic: '{0}'\n"  # 选择合适的模板
        "   by clicking on the 'Add Caption' button below it\n"  # 点击添加标题按钮
        "4. Write a Top Text (setup/context) and Bottom Text (punchline/outcome) related to '{0}'.\n"  # 编写上下文字
        "5. Check the preview making sure it is funny and a meaningful meme. Adjust text directly if needed. \n"  # 检查预览效果
        "6. Look at the meme and text on it, if it doesnt make sense, PLEASE retry by filling the text boxes with different text. \n"  # 如果不合理则重试
        "7. Click on the Generate meme button to generate the meme\n"  # 生成表情包
        "8. Copy the image link and give it as the output\n"  # 复制图片链接作为输出
    ).format(query)

    # 创建浏览器自动化代理
    agent = Agent(
        task=task_description,  # 传入任务描述
        llm=llm,  # 传入选择的语言模型
        max_actions_per_step=5,  # 每步最大操作数，控制执行效率
        max_failures=25,  # 最大失败次数，增强容错性
        use_vision=(model_choice != "Deepseek")  # 除Deepseek外都启用视觉功能
    )

    # 异步运行代理任务
    history = await agent.run()
    
    # 从代理执行历史中提取最终结果
    final_result = history.final_result()
    
    # 使用正则表达式从结果中提取ImgFlip表情包URL
    # 匹配格式: https://imgflip.com/i/[字母数字ID]
    url_match = re.search(r'https://imgflip\.com/i/(\w+)', final_result)
    if url_match:
        # 提取表情包ID并构建直接图片链接
        meme_id = url_match.group(1)
        return f"https://i.imgflip.com/{meme_id}.jpg"  # 返回可直接访问的图片URL
    return None  # 如果没有找到URL则返回None

def main():
    """
    Streamlit应用的主函数，构建用户界面和处理用户交互
    """
    # 自定义CSS样式（当前为空，可以添加样式定制）

    # 设置应用标题和说明
    st.title("🥸 AI Meme Generator Agent - Browser Use")
    st.info("This AI browser agent does browser automation to generate memes based on your input with browser use. Please enter your API key and describe the meme you want to generate.")
    
    # 侧边栏配置区域
    with st.sidebar:
        # 侧边栏标题
        st.markdown('<p class="sidebar-header">⚙️ Model Configuration</p>', unsafe_allow_html=True)
        
        # 模型选择下拉框
        model_choice = st.selectbox(
            "Select AI Model",  # 选择框标签
            ["Claude", "Deepseek", "OpenAI"],  # 可选模型列表
            index=0,  # 默认选择第一个（Claude）
            help="Choose which LLM to use for meme generation"  # 帮助提示
        )
        
        # 根据选择的模型显示对应的API密钥输入框
        api_key = ""
        if model_choice == "Claude":
            # Claude API密钥输入
            api_key = st.text_input("Claude API Key", type="password", 
                                  help="Get your API key from https://console.anthropic.com")
        elif model_choice == "Deepseek":
            # Deepseek API密钥输入
            api_key = st.text_input("Deepseek API Key", type="password",
                                  help="Get your API key from https://platform.deepseek.com")
        else:
            # OpenAI API密钥输入
            api_key = st.text_input("OpenAI API Key", type="password",
                                  help="Get your API key from https://platform.openai.com")

    # 主内容区域
    # 表情包创意输入区域标题
    st.markdown('<p class="header-text">🎨 Describe Your Meme Concept</p>', unsafe_allow_html=True)
    
    # 用户输入表情包创意的文本框
    query = st.text_input(
        "Meme Idea Input",  # 输入框标签
        placeholder="Example: 'Ilya's SSI quietly looking at the OpenAI vs Deepseek debate while diligently working on ASI'",  # 占位符示例
        label_visibility="collapsed"  # 隐藏标签显示
    )

    # 生成表情包按钮及其处理逻辑
    if st.button("Generate Meme 🚀"):
        # 验证API密钥是否已输入
        if not api_key:
            st.warning(f"Please provide the {model_choice} API key")
            st.stop()  # 停止执行
        # 验证表情包创意是否已输入
        if not query:
            st.warning("Please enter a meme idea")
            st.stop()  # 停止执行

        # 显示加载状态并执行表情包生成
        with st.spinner(f"🧠 {model_choice} is generating your meme..."):
            try:
                # 异步调用表情包生成函数
                meme_url = asyncio.run(generate_meme(query, model_choice, api_key))
                
                # 检查是否成功生成表情包
                if meme_url:
                    # 显示成功消息
                    st.success("✅ Meme Generated Successfully!")
                    # 显示生成的表情包图片
                    st.image(meme_url, caption="Generated Meme Preview", use_container_width=True)
                    # 显示图片链接信息
                    st.markdown(f"""
                        **Direct Link:** [Open in ImgFlip]({meme_url})  
                        **Embed URL:** `{meme_url}`
                    """)
                else:
                    # 显示失败消息
                    st.error("❌ Failed to generate meme. Please try again with a different prompt.")
                    
            except Exception as e:
                # 捕获并显示异常错误
                st.error(f"Error: {str(e)}")
                # 提供OpenAI相关的帮助提示
                st.info("💡 If using OpenAI, ensure your account has GPT-4o access")

# 程序入口点
if __name__ == '__main__':
    main()  # 运行主函数