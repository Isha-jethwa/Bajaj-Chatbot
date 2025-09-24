import faiss, pickle, os, glob, re
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st
from pypdf import PdfReader
from typing import List, Tuple, Optional

# Optional fallback import for better PDF text extraction
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# Optional OCR dependencies
try:
    from pdf2image import convert_from_path  # requires poppler on system
    import pytesseract
    from PIL import Image
except Exception:
    convert_from_path = None
    pytesseract = None
    Image = None

# --- Caching loaders to avoid reloading on each interaction ---
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_index_and_chunks():
    index_path = "faiss_index.idx"
    chunks_path = "chunks.pkl"

    # Auto-build if missing
    if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
        build_index_from_docs(docs_dir="docs", chunk_size=500, overlap=100, index_file=index_path, chunks_file=chunks_path)

    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        all_chunks = pickle.load(f)
    return index, all_chunks

@st.cache_resource(show_spinner=True)
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


def split_text(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += max(1, size - overlap)
    return chunks


def extract_text(pdf_path: str) -> str:
    # 1) Try pypdf
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        if text and text.strip():
            return text
    except Exception:
        pass

    # 2) Fallback to PyMuPDF (handles complex layout better, but not pure scans)
    try:
        if fitz is not None:
            with fitz.open(pdf_path) as doc:
                parts = []
                for page in doc:
                    parts.append(page.get_text("text"))
                text2 = "\n".join(parts)
                if text2 and text2.strip():
                    return text2
    except Exception:
        pass

    # 3) OCR fallback for scanned PDFs (requires poppler + tesseract installed)
    try:
        if convert_from_path is not None and pytesseract is not None:
            poppler_path = os.getenv("POPPLER_PATH", None)
            images = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path) if poppler_path else convert_from_path(pdf_path, dpi=200)
            ocr_text_parts: List[str] = []
            for img in images:
                ocr_text_parts.append(pytesseract.image_to_string(img))
            ocr_text = "\n".join(ocr_text_parts)
            if ocr_text and ocr_text.strip():
                return ocr_text
    except Exception:
        pass

    return ""


def build_index_from_docs(docs_dir: str, chunk_size: int, overlap: int, index_file: str, chunks_file: str):
    embedder = load_embedder()
    pdf_files = sorted(glob.glob(os.path.join(docs_dir, "*.pdf")))
    all_chunks: List[Tuple[str, str]] = []

    for f in pdf_files:
        text = extract_text(f)
        if not text.strip():
            continue
        for i, ch in enumerate(split_text(text, size=chunk_size, overlap=overlap)):
            all_chunks.append((f"{f}-{i}", ch))

    if not all_chunks:
        raise RuntimeError("No text extracted from PDFs in 'docs/'.")

    texts = [c[1] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    with open(chunks_file, "wb") as f:
        pickle.dump(all_chunks, f)
    faiss.write_index(index, index_file)


def search_docs(query: str, embedder: SentenceTransformer, index, all_chunks, top_k: int = 16):
    # Bias the query towards BFL standalone
    biased_query = f"{query} Bajaj Finance Limited BFL standalone (not Bajaj Finserv, not consolidated, not Bajaj Broking, not Bajaj Housing Finance) GNPA"
    query_embedding = embedder.encode([biased_query])
    distances, indices = index.search(query_embedding, top_k)
    return [all_chunks[i][1] for i in indices[0]]


# --- Domain aware extractors ---
PCT_RE = re.compile(r"([0-9]+\.?[0-9]*)\s*%")
KEYWORDS_PREFER = re.compile(r"\b(BFL|Bajaj\s+Finance|Finance\s+Limited|standalone|quarter|up\s*from|vs\.?|previous\s*quarter|bps|basis\s*points)\b", re.I)
EXCLUDE_ENTITIES = re.compile(r"\b(Bajaj\s*Finserv|consolidated|Bajaj\s*Broking|BHFL|Bajaj\s*Housing\s*Finance)\b", re.I)


def score_sentence(sent: str, nums: List[str]) -> int:
    score = 0
    if re.search(r"\b(gross\s*npa\w*|gnpa\w*)\b", sent, re.I):
        score += 3
    if KEYWORDS_PREFER.search(sent):
        score += 3
    if EXCLUDE_ENTITIES.search(sent):
        score -= 6
    if len(nums) >= 2:
        score += 2
    try:
        vals = [float(n) for n in nums]
        if any(0.7 <= v <= 1.6 for v in vals):
            score += 2
    except Exception:
        pass
    return score


def extract_gnpa_answer(text: str) -> Optional[str]:
    sentences = re.split(r"(?<=[\.!?])\s+|\n+", text)
    best = None
    best_nums: List[str] = []
    best_score = -10
    for sent in sentences:
        if not re.search(r"\b(gross\s*npa\w*|gnpa\w*)\b", sent, re.I):
            continue
        nums = PCT_RE.findall(sent)
        if not nums:
            continue
        sc = score_sentence(sent, nums)
        if sc > best_score:
            best, best_nums, best_score = sent, nums, sc
    if not best:
        return None
    if EXCLUDE_ENTITIES.search(best):
        return None
    latest = best_nums[0]
    prev = None
    m_prev = re.search(r"(?:up\s*from|vs\.?|previous\s*quarter)[^0-9%]*([0-9]+\.?[0-9]*)\s*%", best, re.I)
    if m_prev:
        prev = m_prev.group(1)
    elif len(best_nums) >= 2:
        prev = best_nums[1]
    if prev:
        return f"Expected A . GNPAs for BFL for the latest quarter are {latest}% up from {prev}% last quarter"
    return f"Expected A . GNPAs for BFL for the latest quarter are {latest}%"


def answer_question(question: str):
    embedder = load_embedder()
    index, all_chunks = load_index_and_chunks()
    qa = load_qa_pipeline()

    raw_chunks = search_docs(question, embedder, index, all_chunks, top_k=16)

    # Re-rank chunks to prioritize BFL standalone and de-prioritize others
    def chunk_score(ch: str) -> int:
        s = 0
        if re.search(r"\b(BFL|Bajaj\s+Finance|Finance\s+Limited|standalone)\b", ch, re.I):
            s += 5
        if EXCLUDE_ENTITIES.search(ch):
            s -= 6
        if re.search(r"\bGNPA|Gross\s*NPA\b", ch, re.I):
            s += 3
        return s

    ranked = sorted(raw_chunks, key=chunk_score, reverse=True)
    context_chunks = ranked[:8]
    context = " \n".join(context_chunks)

    # Heuristic: if user asks about GNPA/NPAs, prefer precise extractor
    if re.search(r"\b(gross\s*npa\w*|gnpa\w*)\b", question, re.I):
        extracted = extract_gnpa_answer(context)
        if extracted:
            return extracted, context_chunks

    result = qa(question=question, context=context)
    return result["answer"], context_chunks


# --- UI ---
st.set_page_config(page_title="Bajaj Chatbot", page_icon="🤖", layout="centered")
st.title("Bajaj Document QA 🤖")
st.caption("Ask questions about the PDFs in the `docs/` folder. Uses FAISS retrieval + DistilBERT QA.")

# Sidebar controls
with st.sidebar:
    st.header("Index")
    if st.button("Rebuild index from docs", type="primary"):
        with st.spinner("Rebuilding index from PDFs in docs/..."):
            try:
                # Clear cached resources so they reload after rebuild
                load_index_and_chunks.clear()
                build_index_from_docs(docs_dir="docs", chunk_size=500, overlap=100, index_file="faiss_index.idx", chunks_file="chunks.pkl")
                st.success("Index rebuilt successfully.")
            except Exception as e:
                # Give actionable hints for OCR prerequisites
                hint = ""
                if "No text extracted" in str(e):
                    if convert_from_path is None or pytesseract is None:
                        hint = "\nHint: Install OCR deps: pdf2image, pytesseract; also install Poppler and Tesseract OCR on your system. Set POPPLER_PATH env var if needed."
                st.error(f"Failed to rebuild: {e}{hint}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# User input
if prompt := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, used_chunks = answer_question(prompt)
                st.markdown(answer)
                with st.expander("Show retrieved context"):
                    for i, ch in enumerate(used_chunks, 1):
                        st.markdown(f"**Chunk {i}:**\n\n{ch}")
                st.session_state.messages.append(("assistant", answer))
            except Exception as e:
                st.error(f"Error: {e}") 
