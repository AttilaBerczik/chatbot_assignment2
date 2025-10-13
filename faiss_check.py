import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_PATH = "faiss_data/faiss_index"
MODEL_NAME = "BAAI/bge-large-en-v1.5"

print("🔍 Checking FAISS vector store at:", FAISS_PATH)

if not os.path.exists(FAISS_PATH):
    raise FileNotFoundError(f"❌ FAISS index not found at {FAISS_PATH}")

# 1️⃣ Load the same embedding model used for ingestion
print("🧠 Loading embedding model:", MODEL_NAME)
embedding = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "cpu"}
)

# 2️⃣ Load the FAISS index
print("📦 Loading FAISS index...")
db = FAISS.load_local(FAISS_PATH, embedding, allow_dangerous_deserialization=True)

# 3️⃣ Check vector dimensionality
faiss_index = db.index
index_dim = faiss_index.d
print(f"✅ FAISS index loaded — dimension: {index_dim}")

# 4️⃣ Sanity check: sample retrieval
test_query = "test"
print(f"🔎 Running a small similarity search with query: '{test_query}'...")
results = db.similarity_search(test_query, k=3)

if not results:
    print("⚠️ No results returned — index may be empty.")
else:
    print(f"✅ Retrieved {len(results)} results.")
    for i, r in enumerate(results):
        print(f"\nResult {i+1}:")
        print("Content preview:", r.page_content[:200].replace("\n", " "))
        print("Metadata:", r.metadata)

print("\n🎉 FAISS index check complete — everything looks good!")
