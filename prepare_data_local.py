from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import faiss, pickle, os
import glob

# --- Configuration ---
DOCS_PATH = "docs/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNKS_OUTPUT_FILE = "chunks.pkl"
FAISS_INDEX_FILE = "faiss_index.idx"

# Load local embedding model
print(f"Loading embedding model: {EMBEDDING_MODEL}...")
embedder = SentenceTransformer(EMBEDDING_MODEL)  # 384 dims

def extract_text(path):
    """Extracts text from a PDF file, handling potential errors."""
    print(f"  Extracting text from: {path}")
    try:
        reader = PdfReader(path)
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        if not text.strip():
            print(f"Warning: No text extracted from {path}. Skipping.")
            return ""
        return text
    except Exception as e:
        print(f"Error reading or extracting text from {path}: {e}")
        return ""

def split_text(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    """Splits text into overlapping chunks."""
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap # Move to the next chunk start
    return chunks

# Find all PDF files in the specified directory
files = glob.glob(os.path.join(DOCS_PATH, "*.pdf"))
if not files:
    print(f"No PDF files found in '{DOCS_PATH}'. Exiting.")
    exit()

print(f"Found {len(files)} PDF files to process.")

all_chunks: List[Tuple[str, str]] = []
for f in files:
    if os.path.exists(f):
        text = extract_text(f)
        if text:
            chunks = split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            for i, ch in enumerate(chunks):
                all_chunks.append((f"{f}-{i}", ch))
    else:
        print(f"Warning: File not found, skipping: {f}")

# Embed
print(f"Embedding {len(all_chunks)} chunks...")
texts = [c[1] for c in all_chunks]
embeddings = embedder.encode(texts, show_progress_bar=True)

# Build FAISS index
print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save everything
with open(CHUNKS_OUTPUT_FILE, "wb") as f:
    pickle.dump(all_chunks, f)
faiss.write_index(index, FAISS_INDEX_FILE)

print(f"✅ Local FAISS index built and saved to '{FAISS_INDEX_FILE}' and '{CHUNKS_OUTPUT_FILE}'.")
