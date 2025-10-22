import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import concurrent.futures
from tqdm import tqdm

# Model paths (same as in app.py - use pre-downloaded models)
MODELS_DIR = os.environ.get("HF_MODELS_DIR", os.path.join(os.getcwd(), "models"))
EMB_LOCAL_DIR = os.path.join(MODELS_DIR, "bge-large-en-v1.5")

def get_internal_links(base_url, limit=10):
    """Extract internal links from any website (same domain only)."""
    try:
        response = requests.get(base_url, headers={"User-Agent": os.environ["USER_AGENT"]}, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {base_url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    domain = urlparse(base_url).netloc
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        # Only keep links within the same domain and HTTP(S)
        if parsed.netloc == domain and parsed.scheme in ["http", "https"]:
            links.add(full_url)

        if len(links) >= limit:
            break

    return list(links)


def crawl_and_embed(base_url, link_limit=10):
    """Crawl a website and create FAISS embeddings."""
    # 1️⃣ Get related internal links
    related_links = get_internal_links(base_url, limit=link_limit)
    urls = [base_url] + related_links

    # 2️⃣ Load all pages
    print(f"Found {len(urls)} URLs to load.")

    def load_one(url):
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            print(f"Loaded {url}")
            return docs
        except Exception as e:
            print(f"Failed to load {url}: {e}")
            return []

    all_documents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(load_one, urls)
        for docs in results:
            all_documents.extend(docs)

    print(f"📚 Loaded {len(all_documents)} total documents.")

    if not all_documents:
        print("No documents loaded. Exiting.")
        return

    print("Loading model and preparing splitter...")

    # Use the cache directory you want
    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    CACHE_DIR = os.path.expanduser("~/chatbot_assignment2/models_cache")

    print("Splitting text into chunks...")

    from transformers import AutoTokenizer
    from tqdm import tqdm
    from langchain.schema import Document

    splitter = SentenceTransformersTokenTextSplitter(
        model_name=MODEL_NAME,
        chunk_size=300,
        chunk_overlap=50,
    )

    texts = []
    for doc in tqdm(all_documents, desc="Splitting docs"):
        texts.extend(splitter.split_documents([doc]))

    tokenizer = splitter.tokenizer
    max_tokens = tokenizer.model_max_length  # usually 512

    # Enforce hard token limit (simple)
    safe_texts = []
    for t in texts:
        ids = tokenizer.encode(t.page_content, add_special_tokens=False)
        if len(ids) <= max_tokens:
            safe_texts.append(t)
        else:
            # break into safe slices with small overlap
            for i in range(0, len(ids), max_tokens - 50):
                piece = ids[i:i + max_tokens]
                text_piece = tokenizer.decode(piece, skip_special_tokens=True)
                safe_texts.append(Document(page_content=text_piece, metadata=t.metadata))

    chunk_token_counts = [len(tokenizer.encode(t.page_content, add_special_tokens=False)) for t in safe_texts]
    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"✂️ Split into {len(safe_texts)} chunks.")
    print(f"🔍 Largest chunk has {max(chunk_token_counts) if chunk_token_counts else 0} tokens.")

    # Use the same model for embeddings, with the same cache
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        cache_folder=CACHE_DIR,
    )

    # 5️⃣ Build and save FAISS index
    print("Build and save FAISS index...")
    db = FAISS.from_documents(texts, embeddings)

# 5️⃣ Convert text into embeddings using the same model as app.py
print(f"Loading embeddings model from {EMB_LOCAL_DIR}...")
embeddings = HuggingFaceEmbeddings(model_name=EMB_LOCAL_DIR, model_kwargs={'device': 'cpu'})

# Create a FAISS vector store from the documents
print("Creating FAISS vector store with optimized indexing...")

# First create basic index to get embeddings
db = FAISS.from_documents(texts, embeddings)

# Get the number of vectors and dimension
n_vectors = db.index.ntotal
d = db.index.d

print(f"Created index with {n_vectors} vectors of dimension {d}")

# Optimize the index if we have enough vectors
if n_vectors > 100:
    # Use IVF (Inverted File) with PQ (Product Quantization) for faster search
    # nlist is the number of clusters (sqrt(n) is a good heuristic)
    nlist = min(int(n_vectors ** 0.5), 100)  # Cap at 100 clusters

    print(f"Optimizing index with IVF-PQ (nlist={nlist})...")

    # Create IVF index with product quantization
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, 8, 8)  # 8 subquantizers, 8 bits each

    # Train the index
    print("Training index...")
    vectors = db.index.reconstruct_n(0, n_vectors)
    index.train(vectors)
    index.add(vectors)

    # Set search parameters (nprobe = number of clusters to search)
    index.nprobe = min(10, nlist)  # Search 10 clusters (or fewer if nlist is small)

    # Replace the index in the FAISS object
    db.index = index
    print(f"✓ Index optimized! Search will query {index.nprobe}/{nlist} clusters")
else:
    print(f"Small dataset ({n_vectors} vectors), using flat index (no optimization needed)")

# Save the vector store to a local file
FAISS_INDEX_PATH = os.path.join("faiss_data", "faiss_index")
os.makedirs("faiss_data", exist_ok=True)
db.save_local(FAISS_INDEX_PATH)

print("Vector store created and saved to faiss_data/faiss_index.")
