# pip install --upgrade langchain langchain-classic langchain-text-splitters langchain-openai langchain-chroma pypdf python-dotenv streamlit
#
# 실행: streamlit run pdf_retriver_st.py

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(page_title="PDF 질문-답변", page_icon="📄")
st.header("📄 PDF 질문-답변")
st.caption("PDF를 업로드하면 내용을 바탕으로 질문에 답해드려요.")


@st.cache_resource(show_spinner="PDF를 읽고 인덱싱하는 중...")
def build_rag_chain(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int):
    """업로드된 PDF 바이트로부터 RAG 체인을 구성한다. 동일 입력은 캐시된다."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        reader = PdfReader(tmp_path)
        pages = [
            Document(page_content=page.extract_text() or "", metadata={"page": i})
            for i, page in enumerate(reader.pages)
        ]
    finally:
        os.unlink(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,        # 하나의 청크가 가질 최대 글자 수
        chunk_overlap=chunk_overlap,  # 청크 간 문맥 연결을 위해 겹칠 글자 수
        length_function=len,          # 길이 측정 기준 (기본 문자열 길이)
        is_separator_regex=False,     # 구분 기호의 정규표현식 해석 여부
    )
    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(texts, embeddings_model)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)

    system_prompt = (
        "너는 질문-답변을 돕는 유능한 비서야. "
        "아래 제공된 맥락(context)만을 사용하여 질문에 답해줘. "
        "답을 모르면 모른다고 하고, 절대 답변을 지어내지 마.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_from_llm, question_answer_chain)


with st.sidebar:
    st.subheader("⚙️ 설정")
    chunk_size = st.slider("청크 크기", 100, 1000, 300, 50)
    chunk_overlap = st.slider("청크 겹침", 0, 200, 20, 10)

uploaded = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded is None:
    st.info("먼저 PDF 파일을 업로드해 주세요.")
    st.stop()

rag_chain = build_rag_chain(uploaded.getvalue(), chunk_size, chunk_overlap)

question = st.text_input("질문", placeholder="예) 먹고 싶어하는 음식은 무엇이야?")

if st.button("질문하기", type="primary") and question:
    with st.spinner("답변을 생성하는 중..."):
        response = rag_chain.invoke({"input": question})

    st.markdown("### 답변")
    st.write(response["answer"])

    context_docs = response.get("context", [])
    st.caption(f"검색된 참조 문서 개수: {len(context_docs)}")
    with st.expander("참조 문서 보기"):
        for i, doc in enumerate(context_docs):
            page = doc.metadata.get("page", "?")
            st.markdown(f"**[{i + 1}] (page {page})**")
            st.write(doc.page_content)
