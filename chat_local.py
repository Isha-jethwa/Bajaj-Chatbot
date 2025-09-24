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


# Robust GNPA extractor
PCT_RE = re.compile(r"([0-9]+\.?[0-9]*)\s*%")

def extract_gnpa_answer(text: str):
    sentences = re.split(r"(?<=[\.!?])\s+|\n+", text)
    for sent in sentences:
        if re.search(r"\b(gross\s*npa\w*|gnpa\w*)\b", sent, re.I):
            nums = PCT_RE.findall(sent)
            if nums:
                latest = nums[0]
                prev = None
                m_prev = re.search(r"(?:up\s*from|vs\.?|previous\s*quarter)[^0-9%]*([0-9]+\.?[0-9]*)\s*%", sent, re.I)
                if m_prev:
                    prev = m_prev.group(1)
                elif len(nums) >= 2:
                    prev = nums[1]
                if prev:
                    return f"GNPAs for BFL for the latest quarter are {latest}% up from {prev}% last quarter"
                return f"GNPAs for BFL for the latest quarter are {latest}%"
    return None


def ask_bot(question):
    chunks = search_docs(question)
    context = " \n".join(chunks)
    if re.search(r"\b(gross\s*npa\w*|gnpa\w*)\b", question, re.I):
        extracted = extract_gnpa_answer(context)
        if extracted:
            return extracted
    result = qa(question=question, context=context)
    return result["answer"]

# Test loop
while True:
    q = input("Ask: ")
    if q.lower() in ["exit","quit"]: break
    a = ask_bot(q)
    print(q + "\n\n" + a)
