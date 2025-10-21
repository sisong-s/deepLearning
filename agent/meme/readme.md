表情包智能体

# 🥸 AI Meme Generator Agent - Browser Use

The AI Meme Generator Agent is a powerful browser automation tool that creates memes using AI agents. This app combines multi-LLM capabilities with automated browser interactions to generate memes based on text prompts through direct website manipulation.

## Features

- **Multi-LLM Support**

  - Claude 3.5 Sonnet (Anthropic)
  - GPT-4o (OpenAI)
  - Deepseek v3 (Deepseek)
  - Automatic model switching with API key validation

- **Browser Automation**:

  - Direct interaction with imgflip.com meme templates
  - Automated search for relevant meme formats
  - Dynamic text insertion for top/bottom captions
  - Image link extraction from generated memes

- **Smart Generation Workflow**:

  - Action verb extraction from prompts
  - Metaphorical template matching
  - Multi-step quality validation
  - Automatic retry mechanism for failed generations

- **User-Friendly Interface**:
  - Model configuration sidebar
  - API key management
  - Direct meme preview with clickable links
  - Responsive error handling

API keys required:

- **Anthropic** (for Claude)
- **Deepseek**
- **OpenAI** (for GPT-4o)

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git
   cd ai_agent_tutorials/ai_meme_generator_browseruse
   ```
2. **Install the dependencies**:

   ```bash

   ```

   Install `playwright` if needed.

   ```bash
   python -m playwright install --with-deps
   ```

   Playwright 是微软开发的一款跨浏览器自动化测试工具，核心功能是模拟用户在浏览器中的操作（如点击、输入、跳转等），同时支持网页内容爬取和自动化任务，目前已成为前端测试和浏览器自动化领域的热门工具。

3. **Run the Streamlit app**:

   ```bash
   streamlit run ai_meme_generator_agent.py

   ```
