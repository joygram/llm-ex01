# pip install -U pypdf langchain-text-splitters
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


reader = PdfReader("unsu.pdf")
pages = [
    Document(page_content=page.extract_text() or "", metadata={"page": i})
    for i, page in enumerate(reader.pages)
]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    separators=["\n\n", " ", ""],
    length_function=len,
    is_separator_regex=False,
)


texts = text_splitter.split_documents(pages)


if texts:
    print("--- [첫 번째 텍스트 조각(Chunk) 객체 출력] ---")
    print(texts[0])

    print("\n--- [첫 번째 조각의 실제 텍스트 내용만 출력] ---")
    print(texts[0].page_content)
else:
    print("분할된 텍스트 조각이 없습니다. PDF 파일 내용을 확인해 주세요.")
