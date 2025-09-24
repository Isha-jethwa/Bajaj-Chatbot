import faiss, pickle, os, glob, re
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st
from pypdf import PdfReader
from typing import List, Tuple, Optional

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
    try:
        reader = PdfReader(pdf_path)
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception:
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


def search_docs(query: str, embedder: SentenceTransformer, index, all_chunks, top_k: int = 5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [all_chunks[i][1] for i in indices[0]]


# --- Domain aware extractors ---
GNPA_PATTERNS = [
    re.compile(r"gross\s*npa\w*\s*(?:for\s+bf[l|i])?.{0,40}?([0-9]+\.?[0-9]*)\s*%", re.I),
    re.compile(r"gnpa\w*.{0,40}?([0-9]+\.?[0-9]*)\s*%", re.I),
]
PREV_PATTERNS = [
    re.compile(r"up\s*from\s*([0-9]+\.?[0-9]*)\s*%", re.I),
    re.compile(r"vs\.?\s*([0-9]+\.?[0-9]*)\s*%", re.I),
    re.compile(r"previous\s*quarter.{0,20}?([0-9]+\.?[0-9]*)\s*%", re.I),
]

def extract_gnpa_answer(text: str) -> Optional[str]:
    latest = None
    prev = None
    for pat in GNPA_PATTERNS:
        m = pat.search(text)
        if m:
            latest = m.group(1)
            break
    for pat in PREV_PATTERNS:
        m = pat.search(text)
        if m:
            prev = m.group(1)
            break
    if latest:
        if prev:
            return f"GNPAs for BFL for the latest quarter are {latest}%, up from {prev}% last quarter."
        return f"GNPAs for BFL for the latest quarter are {latest}%."
    return None


def answer_question(question: str):
    embedder = load_embedder()
    index, all_chunks = load_index_and_chunks()
    qa = load_qa_pipeline()

    context_chunks = search_docs(question, embedder, index, all_chunks, top_k=8)
    context = " \n".join(context_chunks)

    # Heuristic: if user asks about GNPA, prefer precise extractor
    if re.search(r"\b(gross\s*npa|gnpa)\b", question, re.I):
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
                st.error(f"Failed to rebuild: {e}")

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
