import os
import glob
import uuid

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

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
import httpx

LAW_SITES = {
    "kr": "https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EA%B1%B4%EC%B6%95%EB%B2%95",
    "jp": "https://www.japaneselawtranslation.go.jp/en/laws/view/4024/en",
}

DOCS_DIR = "law_docs"
CHROMA_DIR = ".chroma_law"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

os.makedirs(DOCS_DIR, exist_ok=True)

st.set_page_config(
    page_title="건축 법률 자문 도우미",
    page_icon="⚖️",
    layout="wide",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "### 건축 법률 자문 도우미\n국내외 건축법 문서 기반 AI 법률 자문 서비스",
    },
)

# 세션 ID (세션마다 독립 히스토리)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []

# 문서 메타 정보 (표시용)
DOC_META = {
    "건축법_해석례2024.pdf": {"label": "건축법 해석례 2024", "country": "🇰🇷 국내", "lang": "한국어"},
    "건축법_해석례2023.pdf": {"label": "건축법 해석례 2023", "country": "🇰🇷 국내", "lang": "한국어"},
    "Japan_BuildingStandardsLaw_History.pdf": {"label": "Japan Building Standards Law", "country": "🇯🇵 해외", "lang": "English"},
    "Japan_ConstructionLaw2024_Chambers.pdf": {"label": "Japan Construction Law 2024 (Chambers)", "country": "🇯🇵 해외", "lang": "English"},
}


def load_pdf(path: str) -> list[Document]:
    reader = PdfReader(path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": os.path.basename(path), "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]


def get_indexed_sources(db: Chroma) -> set[str]:
    try:
        result = db.get(include=["metadatas"])
        return {m["source"] for m in result["metadatas"] if m and "source" in m}
    except Exception:
        return set()


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_db(status_container=None):
    embeddings_model = get_embeddings()

    if os.path.exists(CHROMA_DIR):
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings_model)
        indexed = get_indexed_sources(db)
    else:
        db = None
        indexed = set()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    newly_indexed = []

    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)
        if fname not in indexed:
            if status_container:
                status_container.write(f"📄 읽는 중: {fname}")
            pages = load_pdf(pdf_path)
            docs = splitter.split_documents(pages)
            if status_container:
                status_container.write(f"🔢 임베딩 중: {fname} ({len(docs)} 청크)")
            if db is None:
                db = Chroma.from_documents(docs, embeddings_model, persist_directory=CHROMA_DIR)
            else:
                db.add_documents(docs)
            if status_container:
                status_container.write(f"✅ 완료: {fname}")
            newly_indexed.append(fname)

    return db, indexed, newly_indexed


@st.cache_resource(show_spinner=False)
def get_retriever(_db):
    return _db.as_retriever(search_kwargs={"k": 6})


def fetch_law_site(url: str, timeout: int = 10) -> str:
    """공식 법률 사이트에서 텍스트 fetch."""
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0"})
        # 태그 제거 후 앞 3000자만
        import re
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:3000]
    except Exception as e:
        return f"(fetch 실패: {e})"


@st.cache_resource(show_spinner=False)
def get_chain(_retriever):
    import re

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """당신은 대한민국 및 일본 건축 법률 전문 자문가입니다.

답변 원칙:
1. 관련 법 조항(제X조 제X항)을 반드시 인용하고 [출처: 파일명 p.X] 형식으로 근거를 표시하세요.
2. 국내(한국)와 일본 법률 비교 질문은 반드시 표(markdown table)로 정리하세요.
3. 공식 사이트 내용에서 최근 개정·변동사항이 있으면 "📌 최근 변동사항" 섹션으로 명시하고 출처 URL을 표기하세요.
4. 근거가 없는 내용은 절대 지어내지 말고 전문가 상담을 권유하세요.
5. 답변 마지막에 "📚 참조 문서"와 "🌐 웹 출처" 섹션을 구분해서 나열하세요.

[참조 문서 내용]
{context}

[공식 법률 사이트]
{web_section}"""),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])

    def retrieve_context(inputs: dict) -> str:
        docs = _retriever.invoke(inputs["question"])
        return "\n\n".join(
            f"[출처: {d.metadata.get('source','')} p.{d.metadata.get('page','')}]\n{d.page_content}"
            for d in docs
        )

    def fetch_web(inputs: dict) -> str:
        kr = fetch_law_site(LAW_SITES["kr"])
        jp = fetch_law_site(LAW_SITES["jp"])
        return (
            f"🇰🇷 국가법령정보센터 ({LAW_SITES['kr']}):\n{kr}\n\n"
            f"🇯🇵 Japanese Law Translation ({LAW_SITES['jp']}):\n{jp}"
        )

    return (
        RunnablePassthrough.assign(
            context=RunnableLambda(retrieve_context),
            web_section=RunnableLambda(fetch_web),
        )
        | prompt
        | llm
        | StrOutputParser()
    )


# ── 데이터 로드 ──────────────────────────────────────────────
with st.status("📚 법률 문서 인덱스 확인 중...", expanded=True) as status:
    db, indexed, newly_indexed = get_db(status_container=status)
    if newly_indexed:
        status.update(label=f"✅ {len(newly_indexed)}개 문서 인덱싱 완료", state="complete", expanded=False)
    else:
        status.update(label="✅ 인덱스 로드 완료", state="complete", expanded=False)

all_sources = sorted(indexed | set(newly_indexed))

# ── 레이아웃 ─────────────────────────────────────────────────
st.title("⚖️ 건축 법률 자문 도우미")
st.caption(f"세션 ID: `{st.session_state.session_id}` · 국내외 건축법 문서 기반 AI 법률 자문")

if db is None:
    st.error(f"`{DOCS_DIR}/` 폴더에 PDF 파일을 넣어주세요.")
    st.stop()

# ── 채팅 체인 ──────────────────────────────────────────────────
retriever = get_retriever(db)
chain = get_chain(retriever)

# ── 인덱스 현황 카드 ─────────────────────────────────────────
domestic = [(s, DOC_META.get(s)) for s in all_sources if DOC_META.get(s, {}).get("country", "").startswith("🇰🇷")]
overseas = [(s, DOC_META.get(s)) for s in all_sources if DOC_META.get(s, {}).get("country", "").startswith("🇯🇵")]
unknown  = [(s, None) for s in all_sources if s not in DOC_META]

with st.expander(f"현재 참조 중인 법률 문서: 국내({len(domestic)}) 해외({len(overseas)})", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🇰🇷 국내 건축법**")
        for fname, meta in domestic:
            tag = "🆕" if fname in newly_indexed else "✅"
            st.markdown(f"{tag} {meta['label'] if meta else fname}")
    with col2:
        st.markdown("**🌏 해외 건축법**")
        for fname, meta in overseas:
            tag = "🆕" if fname in newly_indexed else "✅"
            st.markdown(f"{tag} {meta['label'] if meta else fname}")
    if unknown:
        st.markdown("**📄 기타**")
        for fname, _ in unknown:
            st.markdown(f"{'🆕' if fname in newly_indexed else '✅'} {fname}")

# ── 사이드바 ─────────────────────────────────────────────────
SAMPLE_QUESTIONS = {
    "건축 허가 · 신고": [
        "건축허가와 건축신고의 차이는 무엇인가요?",
        "건축허가 없이 공사를 하면 어떤 처벌을 받나요?",
        "불법 건축물 양성화 조건은 무엇인가요?",
    ],
    "용도 · 면적 기준": [
        "건폐율과 용적률의 차이는 무엇인가요?",
        "용도지역별 건폐율·용적률 한도는 어떻게 되나요?",
        "주거지역에서 상업용도로 변경 가능한가요?",
    ],
    "안전 · 구조 기준": [
        "한국과 일본의 내진 설계 기준을 비교해주세요",
        "일조권 침해 기준은 어떻게 되나요?",
        "건축물 안전점검 의무 주기는 어떻게 되나요?",
    ],
    "국내외 비교": [
        "한국과 일본의 건축허가 절차를 비교해주세요",
        "일본 건축기준법의 용도지역 체계는 한국과 어떻게 다른가요?",
        "한국과 일본의 건축물 높이 제한 기준을 비교해주세요",
        "한국과 일본의 건폐율·용적률 규제를 비교해주세요",
        "한국과 일본의 내진설계 기준 차이는 무엇인가요?",
        "일본은 목조건축 규제가 한국과 어떻게 다른가요?",
    ],
}

with st.sidebar:
    # 벡터 청킹 현황
    try:
        total_chunks = db._collection.count()
        per_source = {}
        result = db.get(include=["metadatas"])
        for m in result["metadatas"]:
            src = m.get("source", "unknown") if m else "unknown"
            per_source[src] = per_source.get(src, 0) + 1
        with st.expander(f"🗂️ 벡터 청킹 현황 ({total_chunks}개)", expanded=False):
            for src, cnt in sorted(per_source.items()):
                meta = DOC_META.get(src)
                label = meta["label"] if meta else src
                country = meta["country"] if meta else "📄"
                st.caption(f"{country} {label}: {cnt}청크")
    except Exception:
        st.caption("청킹 정보 로딩 중...")

    st.divider()

    # 자주 묻는 질문
    st.subheader("💡 자주 묻는 질문")
    for category, questions in SAMPLE_QUESTIONS.items():
        with st.expander(category, expanded=False):
            for q in questions:
                if st.button(q, key=f"sb_{q}", use_container_width=True):
                    st.session_state["pending_question"] = q
                    st.rerun()

    st.divider()
    st.subheader("➕ 문서 추가")
    uploaded = st.file_uploader("PDF 추가 업로드", type="pdf")
    if uploaded:
        save_path = os.path.join(DOCS_DIR, uploaded.name)
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(uploaded.getvalue())
            st.success(f"{uploaded.name} 저장됨.")
            st.cache_resource.clear()
            st.rerun()
        else:
            st.info("이미 존재하는 파일입니다.")

    st.divider()
    if st.button("🗑️ 대화 초기화"):
        st.session_state.messages = []
        st.rerun()

# 자주 묻는 질문 (대화가 없을 때만 표시)
if not st.session_state.messages:
    st.markdown("**💡 자주 묻는 질문**")
    selected_category = st.selectbox(
        "카테고리 선택",
        options=list(SAMPLE_QUESTIONS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    for q in SAMPLE_QUESTIONS[selected_category]:
        if st.button(f"▸ {q}", key=q, use_container_width=True):
            st.session_state["pending_question"] = q
            st.rerun()
    st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 샘플 질문 클릭 처리
pending = st.session_state.pop("pending_question", None)

user_input = st.chat_input("건축 법률에 대해 질문하세요 (국내외 비교 질문도 가능합니다)")

if question := (pending or user_input):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("📄 문서 검색 및 공식 사이트 확인 중..."):
            mid = chain.steps[0].invoke({"question": question, "history": history})

        full = st.write_stream(
            (chain.steps[1] | chain.steps[2]).stream(mid)
        )

    st.session_state.messages.append({"role": "assistant", "content": full})
