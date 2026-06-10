import os
import glob

import streamlit as st
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

DOCS_DIR = "docs"
CHROMA_DIR = ".chroma_company"
os.makedirs(DOCS_DIR, exist_ok=True)

st.set_page_config(page_title="내규 챗봇", page_icon="📋")
st.title("📋 내규 문서 Q&A")


def load_pdf(path: str) -> list[Document]:
    reader = PdfReader(path)
    return [
        Document(page_content=page.extract_text() or "", metadata={"source": os.path.basename(path), "page": i})
        for i, page in enumerate(reader.pages)
    ]


def get_indexed_sources(db: Chroma) -> set[str]:
    try:
        result = db.get(include=["metadatas"])
        return {m["source"] for m in result["metadatas"] if m and "source" in m}
    except Exception:
        return set()


@st.cache_resource(show_spinner=False)
def get_db():
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(CHROMA_DIR):
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings_model)
        indexed = get_indexed_sources(db)
    else:
        db = None
        indexed = set()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20, length_function=len)

    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    new_docs = []
    newly_indexed = []

    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)
        if fname not in indexed:
            pages = load_pdf(pdf_path)
            new_docs.extend(splitter.split_documents(pages))
            newly_indexed.append(fname)

    if new_docs:
        if db is None:
            db = Chroma.from_documents(new_docs, embeddings_model, persist_directory=CHROMA_DIR)
        else:
            db.add_documents(new_docs)

    return db, indexed, newly_indexed


@st.cache_resource(show_spinner=False)
def get_chain(_db):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retriever = _db.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "너는 회사 내규 문서를 기반으로 질문에 답하는 비서야. "
         "아래 맥락만 사용해서 답해줘. 모르면 모른다고 해.\n\n"
         "{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])

    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_docs(retriever.invoke(x["question"])))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# --- DB 로드 ---
with st.spinner("문서 인덱스 확인 중..."):
    db, indexed, newly_indexed = get_db()

if db is None:
    st.warning(f"`{DOCS_DIR}/` 폴더에 PDF 파일을 넣어주세요.")
    st.stop()

# 인덱스 상태 표시
with st.sidebar:
    st.subheader("📂 인덱스된 문서")
    for src in sorted(indexed | set(newly_indexed)):
        if src in newly_indexed:
            st.success(f"✅ {src} (새로 인덱싱)")
        else:
            st.info(f"💾 {src} (이미 참조데이터로 가지고 있습니다)")

    st.divider()
    st.subheader("➕ 문서 추가")
    uploaded = st.file_uploader("PDF 추가 업로드", type="pdf")
    if uploaded:
        save_path = os.path.join(DOCS_DIR, uploaded.name)
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(uploaded.getvalue())
            st.success(f"{uploaded.name} 저장됨. 새로고침하면 인덱싱됩니다.")
            st.cache_resource.clear()
            st.rerun()
        else:
            st.info("이미 존재하는 파일입니다.")

# --- 채팅 ---
chain = get_chain(db)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("내규에 대해 궁금한 점을 질문하세요"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        response_box = st.empty()
        full_response = ""
        for chunk in chain.stream({"question": question, "history": history}):
            full_response += chunk
            response_box.markdown(full_response + "▌")
        response_box.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
