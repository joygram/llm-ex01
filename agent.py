import os, json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

with open("providers.json", "r", encoding="utf-8") as f:
    config = json.load(f)[os.environ["API_PROVIDER"]]

print(f"[*] {config['description']} 모드로 실행합니다. (모델: {os.environ['MODEL_NAME']})")

kwargs = {"temperature": 0}
if config.get("base_url"):
    kwargs["base_url"] = config["base_url"]

real_provider = "google_genai" if os.environ["API_PROVIDER"] == "gemini" else "openai"

llm = init_chat_model(
    model=os.environ["MODEL_NAME"],
    model_provider=real_provider,
    **kwargs,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절하고 명확하게 답변하는 AI 어시스턴트입니다."),
    ("user", "{question}"),
])
chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    answer = chain.invoke({"question": "파이썬이 뭔가요?"})
    print(answer)
