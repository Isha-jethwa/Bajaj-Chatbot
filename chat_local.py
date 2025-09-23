import faiss, pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load embeddings + FAISS
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.idx")
with open("chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

# Load a small local QA model (distilbert)
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def search_docs(query, top_k=3):
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, top_k)
    return [all_chunks[i][1] for i in I[0]]

def ask_bot(question):
    context = " ".join(search_docs(question))
    result = qa(question=question, context=context)
    return result["answer"]

# Test loop
while True:
    q = input("Ask: ")
    if q.lower() in ["exit","quit"]: break
    print("Answer:", ask_bot(q))
