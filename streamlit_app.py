import streamlit as st
import os, json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tabs import python_tutor, chat, lotto


def _get_secret(key):
    return st.secrets.get(key)

API_PROVIDER = _get_secret("API_PROVIDER") or "openai"

for env_key, env_val in [
    ("OPENAI_API_KEY", _get_secret("OPENAI_API_KEY")),
    ("GOOGLE_API_KEY", _get_secret("GOOGLE_API_KEY")),
]:
    if env_val:
        os.environ[env_key] = env_val

with open("providers.json", "r", encoding="utf-8") as f:
    providers = json.load(f)

if API_PROVIDER not in providers:
    st.error(f"❌ API_PROVIDER 값 '{API_PROVIDER}' 이 잘못되었습니다. 사용 가능한 값: {list(providers.keys())}")
    st.stop()

config = providers[API_PROVIDER]
MODEL_NAME = _get_secret("MODEL_NAME") or config["default_model"]


@st.cache_resource
def init_llm(provider, model_name):
    if provider == "llama-cpp":
        from llama_cpp import Llama
        from langchain_core.language_models.llms import LLM
        from typing import Optional, List

        model = Llama(
            model_path=model_name,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )

        class LlamaCppLLM(LLM):
            @property
            def _llm_type(self):
                return "llama-cpp"

            def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
                result = model(prompt, max_tokens=512, stop=stop or [])
                return result["choices"][0]["text"]

        return LlamaCppLLM()
    else:
        kw = {"model": model_name, "temperature": 0}
        if config.get("base_url"):
            kw["base_url"] = config["base_url"]
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_name, temperature=0)
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(**kw)


def _build_chain(system_prompt):
    llm = init_llm(API_PROVIDER, MODEL_NAME)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{question}"),
    ])
    return prompt_template | llm | StrOutputParser()


@st.cache_resource
def get_tutor_chain(_provider, _model):
    return _build_chain(
        "당신은 코딩을 처음 접하는 초등학생에게 파이썬을 가르치는 아주 친절한 선생님입니다. "
        "어려운 전문 용어는 절대 사용하지 말고, 일상적이고 재미있는 비유(장난감, 게임, 학교 생활 등)를 사용하여 쉽게 설명하세요. "
        "말투는 항상 상냥하고 다정하게 존댓말을 사용해 주세요. "
        "코드가 주어졌다면 해당 코드의 역할을 초등학생 눈높이에 맞춰 설명해주세요."
    )


@st.cache_resource
def get_chat_chain(_provider, _model):
    return _build_chain("당신은 친절하고 유능한 AI 어시스턴트입니다. 질문에 명확하고 정확하게 답변해주세요.")


tab1, tab2, tab3 = st.tabs(["파이썬 튜터 (초등학생용)", "일반 채팅", "로또 생성기"])

with tab1:
    python_tutor.render(get_tutor_chain(API_PROVIDER, MODEL_NAME))

with tab2:
    chat.render(get_chat_chain(API_PROVIDER, MODEL_NAME))

with tab3:
    lotto.render()
