"""
AI 健康与健身规划应用程序

这是一个基于 Streamlit 的 Web 应用，使用 Google Gemini AI 模型
为用户生成个性化的饮食和健身计划。

主要功能：
1. 收集用户的基本健康信息
2. 使用 AI 代理生成个性化饮食计划
3. 使用 AI 代理生成个性化健身计划
4. 提供问答系统解答用户疑问

作者：AI Assistant
创建时间：2025年
"""

# 导入必要的库
import streamlit as st  # Streamlit Web 应用框架
from agno.agent import Agent  # AI 代理框架
from agno.models.google import Gemini  # Google Gemini AI 模型

# 配置 Streamlit 页面设置
st.set_page_config(
    page_title="AI Health & Fitness Planner",  # 页面标题
    page_icon="🏋️‍♂️",  # 页面图标
    layout="wide",  # 使用宽布局
    initial_sidebar_state="expanded"  # 侧边栏默认展开
)

# 自定义 CSS 样式
st.markdown("""
    <style>
    /* 主容器样式 */
    .main {
        padding: 2rem;
    }
    
    /* 按钮样式 - 全宽度，圆角，固定高度 */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    
    /* 成功提示框样式 - 绿色背景 */
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0fff4;
        border: 1px solid #9ae6b4;
    }
    
    /* 警告提示框样式 - 橙色背景 */
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fffaf0;
        border: 1px solid #fbd38d;
    }
    
    /* 展开器标题样式 - 加粗字体 */
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

def display_dietary_plan(plan_content):
    """
    显示个性化饮食计划
    
    参数:
        plan_content (dict): 包含饮食计划内容的字典
            - why_this_plan_works: 计划有效的原因
            - meal_plan: 详细的餐食计划
            - important_considerations: 重要注意事项
    """
    # 创建可展开的饮食计划区域
    with st.expander("📋 Your Personalized Dietary Plan", expanded=True):
        # 创建两列布局：左列2/3宽度，右列1/3宽度
        col1, col2 = st.columns([2, 1])
        
        # 左列：显示计划原理和餐食计划
        with col1:
            st.markdown("### 🎯 Why this plan works")
            # 显示计划有效原因，如果没有则显示默认信息
            st.info(plan_content.get("why_this_plan_works", "Information not available"))
            st.markdown("### 🍽️ Meal Plan")
            # 显示详细餐食计划
            st.write(plan_content.get("meal_plan", "Plan not available"))
        
        # 右列：显示重要注意事项
        with col2:
            st.markdown("### ⚠️ Important Considerations")
            # 将注意事项按行分割并逐一显示
            considerations = plan_content.get("important_considerations", "").split('\n')
            for consideration in considerations:
                if consideration.strip():  # 跳过空行
                    st.warning(consideration)

def display_fitness_plan(plan_content):
    """
    显示个性化健身计划
    
    参数:
        plan_content (dict): 包含健身计划内容的字典
            - goals: 健身目标
            - routine: 锻炼例程
            - tips: 专业建议
    """
    # 创建可展开的健身计划区域
    with st.expander("💪 Your Personalized Fitness Plan", expanded=True):
        # 创建两列布局
        col1, col2 = st.columns([2, 1])
        
        # 左列：显示目标和锻炼例程
        with col1:
            st.markdown("### 🎯 Goals")
            # 显示健身目标
            st.success(plan_content.get("goals", "Goals not specified"))
            st.markdown("### 🏋️‍♂️ Exercise Routine")
            # 显示详细锻炼例程
            st.write(plan_content.get("routine", "Routine not available"))
        
        # 右列：显示专业建议
        with col2:
            st.markdown("### 💡 Pro Tips")
            # 将建议按行分割并逐一显示
            tips = plan_content.get("tips", "").split('\n')
            for tip in tips:
                if tip.strip():  # 跳过空行
                    st.info(tip)

def main():
    """
    主函数 - 应用程序的入口点
    
    功能：
    1. 初始化会话状态
    2. 设置页面标题和介绍
    3. 配置 API 密钥
    4. 收集用户信息
    5. 生成个性化计划
    6. 提供问答功能
    """
    
    # 初始化会话状态变量
    # 这些变量在用户会话期间保持数据
    if 'dietary_plan' not in st.session_state:
        st.session_state.dietary_plan = {}  # 存储饮食计划
        st.session_state.fitness_plan = {}  # 存储健身计划
        st.session_state.qa_pairs = []  # 存储问答历史
        st.session_state.plans_generated = False  # 标记是否已生成计划

    # 设置页面主标题
    st.title("🏋️‍♂️ AI Health & Fitness Planner")
    
    # 显示应用介绍信息
    st.markdown("""
        <div style='background-color: #00008B; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        Get personalized dietary and fitness plans tailored to your goals and preferences.
        Our AI-powered system considers your unique profile to create the perfect plan for you.
        </div>
    """, unsafe_allow_html=True)

    # 侧边栏：API 配置区域
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        # API 密钥输入框
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",  # 密码类型，隐藏输入内容
            help="Enter your Gemini API key to access the service"
        )
        
        # 检查是否输入了 API 密钥
        if not gemini_api_key:
            st.warning("⚠️ Please enter your Gemini API Key to proceed")
            st.markdown("[Get your API key here](https://aistudio.google.com/apikey)")
            return  # 如果没有 API 密钥，停止执行
        
        st.success("API Key accepted!")

    # 如果有 API 密钥，继续执行主要功能
    if gemini_api_key:
        try:
            # 初始化 Gemini AI 模型
            gemini_model = Gemini(id="gemini-2.5-flash-preview-05-20", api_key=gemini_api_key)
        except Exception as e:
            st.error(f"❌ Error initializing Gemini model: {e}")
            return

        # 用户资料收集区域
        st.header("👤 Your Profile")
        
        # 创建两列布局收集用户信息
        col1, col2 = st.columns(2)
        
        # 左列：基本信息和活动水平
        with col1:
            # 年龄输入
            age = st.number_input("Age", min_value=10, max_value=100, step=1, help="Enter your age")
            
            # 身高输入
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, step=0.1)
            
            # 活动水平选择
            activity_level = st.selectbox(
                "Activity Level",
                options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
                help="Choose your typical activity level"
            )
            
            # 饮食偏好选择
            dietary_preferences = st.selectbox(
                "Dietary Preferences",
                options=["Vegetarian", "Keto", "Gluten Free", "Low Carb", "Dairy Free"],
                help="Select your dietary preference"
            )

        # 右列：体重、性别和健身目标
        with col2:
            # 体重输入
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, step=0.1)
            
            # 性别选择
            sex = st.selectbox("Sex", options=["Male", "Female", "Other"])
            
            # 健身目标选择
            fitness_goals = st.selectbox(
                "Fitness Goals",
                options=["Lose Weight", "Gain Muscle", "Endurance", "Stay Fit", "Strength Training"],
                help="What do you want to achieve?"
            )

        # 生成计划按钮
        if st.button("🎯 Generate My Personalized Plan", use_container_width=True):
            # 显示加载动画
            with st.spinner("Creating your perfect health and fitness routine..."):
                try:
                    # 创建饮食专家 AI 代理
                    dietary_agent = Agent(
                        name="Dietary Expert",
                        role="Provides personalized dietary recommendations",
                        model=gemini_model,
                        instructions=[
                            "Consider the user's input, including dietary restrictions and preferences.",
                            "Suggest a detailed meal plan for the day, including breakfast, lunch, dinner, and snacks.",
                            "Provide a brief explanation of why the plan is suited to the user's goals.",
                            "Focus on clarity, coherence, and quality of the recommendations.",
                        ]
                    )

                    # 创建健身专家 AI 代理
                    fitness_agent = Agent(
                        name="Fitness Expert",
                        role="Provides personalized fitness recommendations",
                        model=gemini_model,
                        instructions=[
                            "Provide exercises tailored to the user's goals.",
                            "Include warm-up, main workout, and cool-down exercises.",
                            "Explain the benefits of each recommended exercise.",
                            "Ensure the plan is actionable and detailed.",
                        ]
                    )

                    # 整合用户资料信息
                    user_profile = f"""
                    Age: {age}
                    Weight: {weight}kg
                    Height: {height}cm
                    Sex: {sex}
                    Activity Level: {activity_level}
                    Dietary Preferences: {dietary_preferences}
                    Fitness Goals: {fitness_goals}
                    """

                    # 生成饮食计划
                    dietary_plan_response = dietary_agent.run(user_profile)
                    dietary_plan = {
                        "why_this_plan_works": "High Protein, Healthy Fats, Moderate Carbohydrates, and Caloric Balance",
                        "meal_plan": dietary_plan_response.content,
                        "important_considerations": """
                        - Hydration: Drink plenty of water throughout the day
                        - Electrolytes: Monitor sodium, potassium, and magnesium levels
                        - Fiber: Ensure adequate intake through vegetables and fruits
                        - Listen to your body: Adjust portion sizes as needed
                        """
                    }

                    # 生成健身计划
                    fitness_plan_response = fitness_agent.run(user_profile)
                    fitness_plan = {
                        "goals": "Build strength, improve endurance, and maintain overall fitness",
                        "routine": fitness_plan_response.content,
                        "tips": """
                        - Track your progress regularly
                        - Allow proper rest between workouts
                        - Focus on proper form
                        - Stay consistent with your routine
                        """
                    }

                    # 将生成的计划保存到会话状态
                    st.session_state.dietary_plan = dietary_plan
                    st.session_state.fitness_plan = fitness_plan
                    st.session_state.plans_generated = True
                    st.session_state.qa_pairs = []  # 清空之前的问答历史

                    # 显示生成的计划
                    display_dietary_plan(dietary_plan)
                    display_fitness_plan(fitness_plan)

                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")

        # 问答系统 - 只有在生成计划后才显示
        if st.session_state.plans_generated:
            st.header("❓ Questions about your plan?")
            
            # 问题输入框
            question_input = st.text_input("What would you like to know?")

            # 获取答案按钮
            if st.button("Get Answer"):
                if question_input:
                    # 显示加载动画
                    with st.spinner("Finding the best answer for you..."):
                        # 获取当前的计划内容
                        dietary_plan = st.session_state.dietary_plan
                        fitness_plan = st.session_state.fitness_plan

                        # 构建上下文信息
                        context = f"Dietary Plan: {dietary_plan.get('meal_plan', '')}\n\nFitness Plan: {fitness_plan.get('routine', '')}"
                        full_context = f"{context}\nUser Question: {question_input}"

                        try:
                            # 创建通用 AI 代理来回答问题
                            agent = Agent(model=gemini_model, show_tool_calls=True, markdown=True)
                            run_response = agent.run(full_context)

                            # 提取回答内容
                            if hasattr(run_response, 'content'):
                                answer = run_response.content
                            else:
                                answer = "Sorry, I couldn't generate a response at this time."

                            # 将问答对保存到历史记录
                            st.session_state.qa_pairs.append((question_input, answer))
                        except Exception as e:
                            st.error(f"❌ An error occurred while getting the answer: {e}")

            # 显示问答历史
            if st.session_state.qa_pairs:
                st.header("💬 Q&A History")
                # 遍历并显示所有问答对
                for question, answer in st.session_state.qa_pairs:
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")

# 程序入口点
if __name__ == "__main__":
    main()