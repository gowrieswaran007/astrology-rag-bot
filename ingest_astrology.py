"""
Astrology RAG Bot - INGEST program (with OCR)
--------------------------------------------------
Stack: LangChain + Ollama Embeddings + ChromaDB + Tesseract OCR

Some PDF pages have real extractable text. Other pages are SCANNED
IMAGES (no text layer at all) - for those, this script automatically
uses OCR (Tesseract) to "read" the image and convert it to text.

Logic per page:
    1. Try normal text extraction (fast)
    2. If that page has little/no text -> treat it as a scanned image
       -> convert page to an image -> run OCR -> use the OCR text instead

Run this ONCE (or after adding new PDFs to the documents folder):
    python ingest_astrology.py

Requirements (one-time):
    pip install langchain langchain-community langchain-ollama langchain-chroma
    pip install pypdf pdf2image pytesseract pillow

    Also install these SYSTEM tools (not python packages):
    - Tesseract OCR:  https://github.com/UB-Mannheim/tesseract/wiki  (Windows installer)
    - Poppler:        https://github.com/oschwartz10612/poppler-windows/releases
      (after installing poppler, add its \\bin folder to your Windows PATH)
"""

import os
import time
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCS_FOLDER = "documents"
CHROMA_DB_DIR ="chroma_db_astrology"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

MIN_TEXT_LENGTH = 20   # below this -> treat page as scanned, use OCR
OCR_DPI = 100          # lower DPI = less CPU/memory load during OCR


def extract_pdf_with_ocr_fallback(pdf_path, filename):
    """Reads a PDF page by page. Uses normal text extraction where possible,
    falls back to OCR only for pages that have no usable text."""

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    # First pass: figure out which pages need OCR
    native_text = {}
    pages_needing_ocr = []

    for i in range(total_pages):
        text = reader.pages[i].extract_text() or ""
        if len(text.strip()) >= MIN_TEXT_LENGTH:
            native_text[i] = text
        else:
            pages_needing_ocr.append(i)

    print(f"    {total_pages} pages total | "
          f"{len(native_text)} have native text | "
          f"{len(pages_needing_ocr)} need OCR")

    # OCR pass: only convert & OCR the pages that actually need it
    ocr_text = {}
    if pages_needing_ocr:
        print(f"    Running OCR on {len(pages_needing_ocr)} scanned pages "
              f"(this can take a while)...")
        for count, page_num in enumerate(pages_needing_ocr, start=1):
            images = convert_from_path(
                pdf_path,
                dpi=OCR_DPI,
                first_page=page_num + 1,
                last_page=page_num + 1,
            )
            if images:
                ocr_text[page_num] = pytesseract.image_to_string(images[0])

            if count % 25 == 0 or count == len(pages_needing_ocr):
                print(f"      OCR progress: {count}/{len(pages_needing_ocr)} pages")
                time.sleep(2)  # short cooldown pause to reduce CPU heat/load

    # Build final Document list, in page order
    documents = []
    for i in range(total_pages):
        text = native_text.get(i) or ocr_text.get(i) or ""
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": i + 1,
                    "method": "native" if i in native_text else "ocr",
                },
            ))

    return documents


def load_all_pdfs():
    all_docs = []
    for filename in os.listdir(DOCS_FOLDER):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(DOCS_FOLDER, filename)
        print(f"\nProcessing: {filename}")
        docs = extract_pdf_with_ocr_fallback(path, filename)
        all_docs.extend(docs)
        print(f"    Extracted {len(docs)} pages of usable text.")
    return all_docs


def main():
    if not os.path.isdir(DOCS_FOLDER) or not os.listdir(DOCS_FOLDER):
        print(f"[Error] Put your astrology PDF files inside the '{DOCS_FOLDER}' folder first.")
        return

    print("Step 1: Loading PDFs (with OCR fallback for scanned pages)...")
    raw_docs = load_all_pdfs()

    if not raw_docs:
        print("No usable text extracted from any PDF.")
        return

    print(f"\nStep 2: Splitting {len(raw_docs)} pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"    Created {len(chunks)} chunks.")

    print(f"\nStep 3: Creating embeddings with '{EMBED_MODEL}' and storing in ChromaDB...")
    print("    (This may take a while for large books - please wait)")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Batch in groups to avoid overwhelming Ollama on very large books
    batch_size = 50
    vectorstore = None
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_DB_DIR,
            )
        else:
            vectorstore.add_documents(batch)
        print(f"    Embedded {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")
        time.sleep(1)  # short cooldown pause between batches

    print(f"\n✅ Done! {len(chunks)} chunks embedded and saved to '{CHROMA_DB_DIR}/'")
    print("Now run: python retrieve_astrology.py")


if __name__ == "__main__":
    main()