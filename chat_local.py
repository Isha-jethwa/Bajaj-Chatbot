import faiss, pickle, re
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load embeddings + FAISS
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.idx")
with open("chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

# Load a small local QA model (distilbert)
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


def search_docs(query, top_k=8):
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, top_k)
    return [all_chunks[i][1] for i in I[0]]


# Domain-aware GNPA extractor
GNPA_PATTERNS = [
    re.compile(r"gross\s*npa\w*\s*(?:for\s+bf[l|i])?.{0,40}?([0-9]+\.?[0-9]*)\s*%", re.I),
    re.compile(r"gnpa\w*.{0,40}?([0-9]+\.?[0-9]*)\s*%", re.I),
]
PREV_PATTERNS = [
    re.compile(r"up\s*from\s*([0-9]+\.?[0-9]*)\s*%", re.I),
    re.compile(r"vs\.?\s*([0-9]+\.?[0-9]*)\s*%", re.I),
    re.compile(r"previous\s*quarter.{0,20}?([0-9]+\.?[0-9]*)\s*%", re.I),
]

def extract_gnpa_answer(text: str):
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


def ask_bot(question):
    chunks = search_docs(question)
    context = " \n".join(chunks)
    if re.search(r"\b(gross\s*npa|gnpa)\b", question, re.I):
        extracted = extract_gnpa_answer(context)
        if extracted:
            return extracted
    result = qa(question=question, context=context)
    return result["answer"]

# Test loop
while True:
    q = input("Ask: ")
    if q.lower() in ["exit","quit"]: break
    print("Answer:", ask_bot(q))
