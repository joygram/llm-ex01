import streamlit as st
import os, json
from langchain.chat_models import init_chat_model

# st.secrets 우선, 없으면 .env 폴백 (로컬 개발용)
def _get_secret(key):
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        from dotenv import load_dotenv
        load_dotenv(override=True)
        return os.environ.get(key)

API_PROVIDER = _get_secret("API_PROVIDER") or "openai"

for env_key, env_val in [
    ("OPENAI_API_KEY",  _get_secret("OPENAI_API_KEY")),
    ("GOOGLE_API_KEY",  _get_secret("GOOGLE_API_KEY")),
]:
    if env_val:
        os.environ[env_key] = env_val

with open("providers.json", "r", encoding="utf-8") as f:
    config = json.load(f)[API_PROVIDER]

MODEL_NAME = _get_secret("MODEL_NAME") or config["default_model"]

# 추가 옵션 세팅 (base_url 등)
kwargs = {"temperature": 0}
if config.get("base_url"):
    kwargs["base_url"] = config["base_url"]

real_provider = "google_genai" if API_PROVIDER == "gemini" else "openai"

llm = init_chat_model(
    model=MODEL_NAME,
    model_provider=real_provider,
    **kwargs
)

tab1, tab2 = st.tabs(["파이썬 튜터 (초등학생용)", "로또 생성기"])

with tab1:
    st.header("🐍 초등학생도 이해하는 파이썬 리뷰")
    st.markdown("파이썬에 대해 궁금한 점을 질문해보세요. 알기 쉽게 설명해 드립니다!")
    
    from streamlit_ace import st_ace
    
    st.markdown("**1. 파이썬 코드 붙여넣기 (선택사항)**")
    code_input = st_ace(language="python", theme="monokai", height=200, key="python_code", auto_update=True)
    
    st.markdown("**2. 궁금한 점 질문하기**")
    question_input = st.text_area("위 코드에 대한 질문이나 파이썬 개념에 대해 궁금한 점을 적어주세요:", key="python_q")
    
    if st.button("질문하기", key="python_btn"):
        if code_input.strip() or question_input.strip():
            prompt_text = ""
            if code_input.strip():
                prompt_text += f"[사용자 코드]\n```python\n{code_input}\n```\n\n"
            if question_input.strip():
                prompt_text += f"[질문 내용]\n{question_input}\n\n"
                
            messages = [
                ("system", """당신은 코딩을 처음 접하는 초등학생에게 파이썬을 가르치는 아주 친절한 선생님입니다.
어려운 전문 용어는 절대 사용하지 말고, 일상적이고 재미있는 비유(장난감, 게임, 학교 생활 등)를 사용하여 쉽게 설명하세요.
말투는 항상 상냥하고 다정하게 존댓말을 사용해 주세요. 코드가 주어졌다면 해당 코드의 역할을 초등학생 눈높이에 맞춰 설명해주세요."""),
                ("user", f"{prompt_text}위 내용에 대해서 초등학생도 완전히 이해할 수 있도록 아주 쉽고 친절하게 설명해줘.")
            ]
            
            st.markdown("### 선생님의 답변:")
            response_container = st.empty()
            full_response = ""
            for chunk in llm.stream(messages):
                full_response += chunk.content
                response_container.markdown(full_response + "▌")
            response_container.markdown(full_response)
        else:
            st.warning("코드나 질문 중 하나는 꼭 입력해주세요!")

with tab2:
    st.header("🍀 로또 번호 생성기")
    st.markdown("행운의 로또 번호 6자리를 생성해보세요!")
    
    import random
    import json
    import os
    from datetime import datetime
    
    @st.cache_data
    def load_lotto_history():
        if os.path.exists("lotto_data.json"):
            with open("lotto_data.json", "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
        
    past_draws = load_lotto_history()
    
    if "lotto_history" not in st.session_state:
        st.session_state["lotto_history"] = []
        
    if st.button("번호 생성", key="lotto_btn"):
        lotto_numbers = sorted(random.sample(range(1, 46), 6))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 과거 모든 당첨 내역 누적 확인 (보너스 번호 제외)
        history_stats = {1: 0, 3: 0, 4: 0, 5: 0}
        
        for draw_no, numbers in past_draws.items():
            match_count = len(set(lotto_numbers) & set(numbers))
            if match_count == 6:
                history_stats[1] += 1
            elif match_count == 5:
                history_stats[3] += 1
            elif match_count == 4:
                history_stats[4] += 1
            elif match_count == 3:
                history_stats[5] += 1
                
        total_wins = sum(history_stats.values())
                
        st.session_state["lotto_history"].insert(0, {
            "time": now, 
            "numbers": lotto_numbers,
            "history_stats": history_stats,
            "total_wins": total_wins
        })
        
    if st.session_state["lotto_history"]:
        for idx, record in enumerate(st.session_state["lotto_history"]):
            total_wins = record.get("total_wins", 0)
            if total_wins > 0:
                stats = record["history_stats"]
                summary = []
                if stats[1] > 0: summary.append(f"1등 {stats[1]}회")
                if stats[3] > 0: summary.append(f"3등(또는 2등) {stats[3]}회")
                if stats[4] > 0: summary.append(f"4등 {stats[4]}회")
                if stats[5] > 0: summary.append(f"5등 {stats[5]}회")
                
                st.markdown(f"🕒 **생성 일시:** {record['time']} | 🎯 **역대 당첨 이력 총 {total_wins}회!** ({', '.join(summary)})")
            else:
                st.caption(f"🕒 생성 일시: {record['time']} | 😢 역대 당첨 내역 없음 (한 번도 당첨된 적 없는 번호)")
            
            html_content = "<div style='display: flex; gap: 15px; margin-top: 5px; margin-bottom: 10px;'>"
            for num in record["numbers"]:
                if num <= 10: color = "#fbc400"
                elif num <= 20: color = "#69c8f2"
                elif num <= 30: color = "#ff7272"
                elif num <= 40: color = "#aaaaaa"
                else: color = "#b0d840"
                
                html_content += f"<div style='width: 50px; height: 50px; border-radius: 50%; background-color: {color}; color: white; display: flex; justify-content: center; align-items: center; font-size: 20px; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); box-shadow: 2px 2px 5px rgba(0,0,0,0.2);'>{num}</div>"
            html_content += "</div>"
            
            st.markdown(html_content, unsafe_allow_html=True)
            
            if idx < len(st.session_state["lotto_history"]) - 1:
                st.divider()
