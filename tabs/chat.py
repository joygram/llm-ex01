import streamlit as st


def render(chain):
    st.header("💬 일반 채팅")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("무엇이든 물어보세요..."):
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            for chunk in chain.stream({"question": user_input}):
                full_response += chunk
                response_container.markdown(full_response + "▌")
            response_container.markdown(full_response)

        st.session_state["chat_messages"].append({"role": "assistant", "content": full_response})
