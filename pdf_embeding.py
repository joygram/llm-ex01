# pip install --upgrade langchain langchain-community langchain-text-splitters langchain-openai langchain-chroma pypdf python-dotenv

from dotenv import load_dotenv
load_dotenv()

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

reader = PdfReader("unsu.pdf")
pages = [
    Document(page_content=page.extract_text() or "", metadata={"page": i})
    for i, page in enumerate(reader.pages)
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,           # 하나의 청크가 가질 최대 글자 수
    chunk_overlap = 20,         # 청크 간에 겹칠 글자 수 (문맥 단절 방지)
    length_function = len,      # 길이를 측정할 함수 (기본 문자열 길이)
    is_separator_regex = False, # 구분 기호(separator)를 정규표현식으로 해석할지 여부
)

texts = text_splitter.split_documents(pages)

embeddings_model = OpenAIEmbeddings()

db = Chroma.from_documents(texts, embeddings_model)
print(db)