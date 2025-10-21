# 导入必要的库和模块
from agno.agent import Agent  # Agno框架的智能代理类
from agno.models.google import Gemini  # Google Gemini模型接口
from agno.media import Image as AgnoImage  # Agno框架的图像处理类
from agno.tools.duckduckgo import DuckDuckGoTools  # DuckDuckGo搜索工具
import streamlit as st  # Streamlit Web应用框架
from typing import List, Optional  # 类型提示支持
import logging  # 日志记录模块
from pathlib import Path  # 路径处理模块
import tempfile  # 临时文件处理
import os  # 操作系统接口

# Configure logging for errors only
# 配置日志记录，仅记录错误级别的信息
# logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def initialize_agents(api_key: str) -> tuple[Agent, Agent, Agent, Agent]:
    """
    初始化四个不同功能的AI代理
    
    参数:
        api_key (str): Google Gemini API密钥
    
    返回:
        tuple: 包含四个代理的元组 (治疗师代理, 闭合代理, 日程规划代理, 直言代理)
    """
    try:
        # 使用提供的API密钥初始化Gemini模型
        model = Gemini(id="gemini-2.0-flash-exp", api_key=api_key)
        
        # 创建治疗师代理 - 提供情感支持和共情
        therapist_agent = Agent(
            model=model,
            name="Therapist Agent",  # 代理名称
            instructions=[
                "You are an empathetic therapist that:",  # 你是一个富有同理心的治疗师
                "1. Listens with empathy and validates feelings",  # 1. 用同理心倾听并验证感受
                "2. Uses gentle humor to lighten the mood",  # 2. 使用温和的幽默来缓解情绪
                "3. Shares relatable breakup experiences",  # 3. 分享相关的分手经历
                "4. Offers comforting words and encouragement",  # 4. 提供安慰的话语和鼓励
                "5. Analyzes both text and image inputs for emotional context",  # 5. 分析文本和图像输入的情感背景
                "Be supportive and understanding in your responses"  # 在回应中要支持和理解
            ],
            markdown=True  # 启用Markdown格式输出
        )

        # 创建闭合代理 - 帮助用户获得情感闭合
        closure_agent = Agent(
            model=model,
            name="Closure Agent",  # 代理名称
            instructions=[
                "You are a closure specialist that:",  # 你是一个闭合专家
                "1. Creates emotional messages for unsent feelings",  # 1. 为未表达的感情创建情感信息
                "2. Helps express raw, honest emotions",  # 2. 帮助表达原始、诚实的情感
                "3. Formats messages clearly with headers",  # 3. 用标题清晰地格式化信息
                "4. Ensures tone is heartfelt and authentic",  # 4. 确保语调真诚和真实
                "Focus on emotional release and closure"  # 专注于情感释放和闭合
            ],
            markdown=True  # 启用Markdown格式输出
        )

        # 创建日程规划代理 - 制定恢复计划
        routine_planner_agent = Agent(
            model=model,
            name="Routine Planner Agent",  # 代理名称
            instructions=[
                "You are a recovery routine planner that:",  # 你是一个恢复日程规划师
                "1. Designs 7-day recovery challenges",  # 1. 设计7天恢复挑战
                "2. Includes fun activities and self-care tasks",  # 2. 包括有趣的活动和自我护理任务
                "3. Suggests social media detox strategies",  # 3. 建议社交媒体排毒策略
                "4. Creates empowering playlists",  # 4. 创建赋权播放列表
                "Focus on practical recovery steps"  # 专注于实用的恢复步骤
            ],
            markdown=True  # 启用Markdown格式输出
        )

        # 创建直言代理 - 提供客观直接的反馈
        brutal_honesty_agent = Agent(
            model=model,
            name="Brutal Honesty Agent",  # 代理名称
            tools=[DuckDuckGoTools()],  # 配备DuckDuckGo搜索工具
            instructions=[
                "You are a direct feedback specialist that:",  # 你是一个直接反馈专家
                "1. Gives raw, objective feedback about breakups",  # 1. 对分手给出原始、客观的反馈
                "2. Explains relationship failures clearly",  # 2. 清楚地解释关系失败的原因
                "3. Uses blunt, factual language",  # 3. 使用直率、事实性的语言
                "4. Provides reasons to move forward",  # 4. 提供前进的理由
                "Focus on honest insights without sugar-coating"  # 专注于诚实的见解，不加粉饰
            ],
            markdown=True  # 启用Markdown格式输出
        )
        
        # 返回所有四个代理
        return therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent
    except Exception as e:
        # 如果初始化失败，显示错误信息并返回None
        st.error(f"Error initializing agents: {str(e)}")
        return None, None, None, None

# Set page config and UI elements
# 设置页面配置和UI元素
st.set_page_config(
    page_title="💔 Breakup Recovery Squad",  # 页面标题
    page_icon="💔",  # 页面图标
    layout="wide"  # 使用宽布局
)



# Sidebar for API key input
# 侧边栏用于API密钥输入
with st.sidebar:
    st.header("🔑 API Configuration")  # 侧边栏标题

    # 初始化会话状态中的API密钥输入
    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = ""
        
    # API密钥输入框
    api_key = st.text_input(
        "Enter your Gemini API Key",  # 输入框标签
        value=st.session_state.api_key_input,  # 当前值
        type="password",  # 密码类型，隐藏输入内容
        help="Get your API key from Google AI Studio",  # 帮助提示
        key="api_key_widget"  # 组件唯一标识
    )

    # 更新会话状态中的API密钥
    if api_key != st.session_state.api_key_input:
        st.session_state.api_key_input = api_key
    
    # 根据API密钥状态显示不同的提示信息
    if api_key:
        st.success("API Key provided! ✅")  # 成功提示
    else:
        st.warning("Please enter your API key to proceed")  # 警告提示
        # 显示获取API密钥的指导信息
        st.markdown("""
        To get your API key:
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Enable the Generative Language API in your [Google Cloud Console](https://console.developers.google.com/apis/api/generativelanguage.googleapis.com)
        """)

# Main content
# 主要内容区域
st.title("💔 Breakup Recovery Squad")  # 主标题
st.markdown("""
    ### Your AI-powered breakup recovery team is here to help!
    Share your feelings and chat screenshots, and we'll help you navigate through this tough time.
""")  # 应用描述

# Input section
# 输入区域
col1, col2 = st.columns(2)  # 创建两列布局

# 左列：文字输入
with col1:
    st.subheader("Share Your Feelings")  # 子标题
    user_input = st.text_area(
        "How are you feeling? What happened?",  # 文本区域标签
        height=150,  # 高度设置
        placeholder="Tell us your story..."  # 占位符文本
    )
    
# 右列：图片上传
with col2:
    st.subheader("Upload Chat Screenshots")  # 子标题
    uploaded_files = st.file_uploader(
        "Upload screenshots of your chats (optional)",  # 文件上传器标签
        type=["jpg", "jpeg", "png"],  # 允许的文件类型
        accept_multiple_files=True,  # 允许多文件上传
        key="screenshots"  # 组件唯一标识
    )
    
    # 如果有上传的文件，显示预览
    if uploaded_files:
        for file in uploaded_files:
            st.image(file, caption=file.name, use_container_width=True)

# Process button and API key check
# 处理按钮和API密钥检查
if st.button("Get Recovery Plan 💝", type="primary"):  # 主要按钮
    # 检查是否提供了API密钥
    if not st.session_state.api_key_input:
        st.warning("Please enter your API key in the sidebar first!")
    else:
        # 初始化所有代理
        therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent = initialize_agents(st.session_state.api_key_input)
        
        # 检查所有代理是否成功初始化
        if all([therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent]):
            # 检查是否有用户输入或上传的文件
            if user_input or uploaded_files:
                try:
                    st.header("Your Personalized Recovery Plan")  # 个性化恢复计划标题
                    
                    def process_images(files):
                        """
                        处理上传的图片文件
                        
                        参数:
                            files: 上传的文件列表
                        
                        返回:
                            list: 处理后的AgnoImage对象列表
                        """
                        processed_images = []
                        for file in files:
                            try:
                                # 创建临时文件路径
                                temp_dir = tempfile.gettempdir()
                                temp_path = os.path.join(temp_dir, f"temp_{file.name}")
                                
                                # 将上传的文件写入临时文件
                                with open(temp_path, "wb") as f:
                                    f.write(file.getvalue())
                                
                                # 创建AgnoImage对象
                                agno_image = AgnoImage(filepath=Path(temp_path))
                                processed_images.append(agno_image)
                                
                            except Exception as e:
                                # 记录图片处理错误
                                logger.error(f"Error processing image {file.name}: {str(e)}")
                                continue
                        return processed_images
                    
                    # 处理所有上传的图片
                    all_images = process_images(uploaded_files) if uploaded_files else []
                    
                    # Therapist Analysis
                    # 治疗师分析
                    with st.spinner("🤗 Getting empathetic support..."):  # 显示加载状态
                        # 构建治疗师提示词
                        therapist_prompt = f"""
                        Analyze the emotional state and provide empathetic support based on:
                        User's message: {user_input}
                        
                        Please provide a compassionate response with:
                        1. Validation of feelings
                        2. Gentle words of comfort
                        3. Relatable experiences
                        4. Words of encouragement
                        """
                        
                        # 运行治疗师代理
                        response = therapist_agent.run(
                            message=therapist_prompt,
                            images=all_images
                        )
                        
                        # 显示治疗师的回应
                        st.subheader("🤗 Emotional Support")
                        st.markdown(response.content)
                    
                    # Closure Messages
                    # 闭合信息
                    with st.spinner("✍️ Crafting closure messages..."):  # 显示加载状态
                        # 构建闭合代理提示词
                        closure_prompt = f"""
                        Help create emotional closure based on:
                        User's feelings: {user_input}
                        
                        Please provide:
                        1. Template for unsent messages
                        2. Emotional release exercises
                        3. Closure rituals
                        4. Moving forward strategies
                        """
                        
                        # 运行闭合代理
                        response = closure_agent.run(
                            message=closure_prompt,
                            images=all_images
                        )
                        
                        # 显示闭合建议
                        st.subheader("✍️ Finding Closure")
                        st.markdown(response.content)
                    
                    # Recovery Plan
                    # 恢复计划
                    with st.spinner("📅 Creating your recovery plan..."):  # 显示加载状态
                        # 构建日程规划代理提示词
                        routine_prompt = f"""
                        Design a 7-day recovery plan based on:
                        Current state: {user_input}
                        
                        Include:
                        1. Daily activities and challenges
                        2. Self-care routines
                        3. Social media guidelines
                        4. Mood-lifting music suggestions
                        """
                        
                        # 运行日程规划代理
                        response = routine_planner_agent.run(
                            message=routine_prompt,
                            images=all_images
                        )
                        
                        # 显示恢复计划
                        st.subheader("📅 Your Recovery Plan")
                        st.markdown(response.content)
                    
                    # Honest Feedback
                    # 诚实反馈
                    with st.spinner("💪 Getting honest perspective..."):  # 显示加载状态
                        # 构建直言代理提示词
                        honesty_prompt = f"""
                        Provide honest, constructive feedback about:
                        Situation: {user_input}
                        
                        Include:
                        1. Objective analysis
                        2. Growth opportunities
                        3. Future outlook
                        4. Actionable steps
                        """
                        
                        # 运行直言代理
                        response = brutal_honesty_agent.run(
                            message=honesty_prompt,
                            images=all_images
                        )
                        
                        # 显示诚实观点
                        st.subheader("💪 Honest Perspective")
                        st.markdown(response.content)
                            
                except Exception as e:
                    # 记录分析过程中的错误
                    logger.error(f"Error during analysis: {str(e)}")
                    st.error("An error occurred during analysis. Please check the logs for details.")
            else:
                # 提示用户需要输入内容
                st.warning("Please share your feelings or upload screenshots to get help.")
        else:
            # 代理初始化失败的错误提示
            st.error("Failed to initialize agents. Please check your API key.")

# Footer
# 页脚
st.markdown("---")  # 分隔线
st.markdown("""
    <div style='text-align: center'>
        <p>Made with ❤️ by the Breakup Recovery Squad</p>
        <p>Share your recovery journey with #BreakupRecoverySquad</p>
    </div>
""", unsafe_allow_html=True)  # 居中显示的页脚信息