import streamlit as st
import os, json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 수엉 
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI


load_dotenv()

with open("providers.json", "r", encoding="utf-8") as f:
    config = json.load(f)[os.environ["API_PROVIDER"]]

print(f"[*] {config['description']} 모드로 실행합니다. (모델: {os.environ['MODEL_NAME']})")

# 추가 옵션 세팅 (base_url 등)
kwargs = {"temperature": 0}
if config.get("base_url"):
    kwargs["base_url"] = config["base_url"]

# LangChain의 내장 팩토리 함수(init_chat_model) 사용!
# gemini는 google_genai로, 나머지는 openai 규격으로 통일
real_provider = "google_genai" if os.environ["API_PROVIDER"] == "gemini" else "openai"

llm = init_chat_model(
    model=os.environ["MODEL_NAME"],
    model_provider=real_provider,
    **kwargs
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 친절하고 명확하게 답변하는 AI 어시스턴트입니다. 
    - 한자 표시 금지.
    """), 
    ("user", "{question}")
])
chain = prompt | llm | StrOutputParser()

# 2. 최신 채팅 방식 (권장)
chat_model = ChatOpenAI(
    base_url="https://api.blackbox.ai/v1", 
    model="blackboxai/minimax/minimax-free"
)



subject = st.text_input("무엇이 궁금해요?")

if st.button("go"):
    messages = [
        ("system", """당신은 '한국어(Korean)'로만 소통하는 하녀입니다.
내부적으로 중국어 데이터가 있거나 중국어로 생각하더라도, 주인님께 말씀드릴 때는 **반드시 100% 한국어로 자연스럽게 번역해서** 답변해야 합니다.
중국어(한자, 병음) 원문은 절대 출력하지 마세요.
모르는 내용일 경우에도 중국어를 쓰지 말고, "주인님, 그 부분은 제가 잘 모르겠습니다."라고 한국어로만 답변하세요."""),
        ("user", f"사용자 질문\n---\n{subject}\n---\n위 내용에 대해서 자세히 알려줘. (경고: 무조건 한국어로만 작성할 것)")
    ]
    
    with st.spinner("Wait for it..."):
        response = chat_model.invoke(messages)
        
    st.write(response.content)

