import streamlit as st
from streamlit_ace import st_ace


def render(chain):
    st.header("🐍 초등학생도 이해하는 파이썬 리뷰")
    st.markdown("파이썬에 대해 궁금한 점을 질문해보세요. 알기 쉽게 설명해 드립니다!")

    st.markdown("**1. 파이썬 코드 붙여넣기 (선택사항)**")
    code_input = st_ace(language="python", theme="monokai", height=200, key="python_code", auto_update=True)

    st.markdown("**2. 궁금한 점 질문하기**")
    question_input = st.text_area("위 코드에 대한 질문이나 파이썬 개념에 대해 궁금한 점을 적어주세요:", key="python_q")

    if st.button("질문하기", key="python_btn"):
        if code_input.strip() or question_input.strip():
            question = ""
            if code_input.strip():
                question += f"[사용자 코드]\n```python\n{code_input}\n```\n\n"
            if question_input.strip():
                question += f"[질문 내용]\n{question_input}\n\n"
            question += "위 내용에 대해서 초등학생도 완전히 이해할 수 있도록 아주 쉽고 친절하게 설명해줘."

            st.markdown("### 선생님의 답변:")
            response_container = st.empty()
            full_response = ""
            for chunk in chain.stream({"question": question}):
                full_response += chunk
                response_container.markdown(full_response + "▌")
            response_container.markdown(full_response)
        else:
            st.warning("코드나 질문 중 하나는 꼭 입력해주세요!")
