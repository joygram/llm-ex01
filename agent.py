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
    ("system", "당신은 친절하고 명확하게 답변하는 AI 어시스턴트입니다."), 
    ("user", "{question}")
])
chain = prompt | llm | StrOutputParser()


# ==========================================
# 수업: 수동으로 객체를 찍어낼 때의 올바른 예시
# ==========================================
# 환경변수 OPENAI_API_KEY는 자동으로 읽어오지만, 
# 목적지가 Blackbox이므로 base_url과 model을 콕 집어줘야 합니다.

# 2. 최신 채팅 방식 (권장)
chat_model = ChatOpenAI(
    base_url="https://api.blackbox.ai/v1", 
    model="blackboxai/minimax/minimax-free"
)

if __name__ == "__main__":
    subject = "날씨"
    response = chat_model.invoke(f"{subject}에 대해서 알려줘")
    print(response.content)
