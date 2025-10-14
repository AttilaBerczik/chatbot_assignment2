import os
import multiprocessing

import vectors

multiprocessing.set_start_method("spawn", force=True)
from huggingface_hub import snapshot_download

# ------------------ CACHE SETUP ------------------
CACHE_DIR = os.path.join(os.getcwd(), "models_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

MODEL_NAME = "BAAI/bge-large-en-v1.5"

local_model_path = os.path.join(CACHE_DIR, MODEL_NAME.replace("/", "__"))
if not os.path.exists(local_model_path):
    print(f"Downloading model {MODEL_NAME} to local cache...")
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=local_model_path,
        local_dir_use_symlinks=False
    )
else:
    print(f"Using cached model from {local_model_path}")

# ------------------ IMPORTS ------------------
import requests
import concurrent.futures
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
import subprocess
import re
from tqdm import tqdm
import faiss
import numpy as np

# ---------------- GPU DETECTION ----------------
def get_free_gpus(min_free_mem_gb=35.0):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
        )
        free_memories = [int(x) for x in result.stdout.strip().split("\n")]
        free_gpus = [i for i, mem in enumerate(free_memories) if mem > min_free_mem_gb * 1024]
        return free_gpus
    except Exception as e:
        print("Could not check GPU memory:", e)
        return list(range(torch.cuda.device_count()))

def get_device():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        best_gpu = max(range(num_gpus), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
        return f"cuda:{best_gpu}"
    else:
        print("No GPU available — using CPU instead.")
        return "cpu"

device = get_device()

# ---------------- SETTINGS ----------------
START_URL = "https://en.wikipedia.org/wiki/Norway"
MAX_LINKS = 10
MAX_WORKERS = 20
os.environ["USER_AGENT"] = "FastCrawlerBot/1.0 (+https://example.com)"

# ---------------- HELPERS ----------------
def get_site_links(base_url, html, limit=1000):
    soup = BeautifulSoup(html, "lxml")
    links = set()
    base_domain = urlparse(base_url).netloc
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        parsed = urlparse(href)
        if parsed.netloc == base_domain and href.startswith("http"):
            links.add(href)
        if len(links) >= limit:
            break
    return list(links)

def fetch(url, headers):
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        print(f"Loaded: {url}")
        return url, resp.text
    except Exception as e:
        print(f"Failed: {url} -> {e}")
        return url, None

def parallel_fetch(urls, headers, max_workers=10):
    pages = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch, url, headers): url for url in urls}
        for future in concurrent.futures.as_completed(futures):
            url, content = future.result()
            if content:
                pages.append((url, content))
    return pages

from bs4 import BeautifulSoup
from langchain_core.documents import Document

def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "footer", "header", "nav", "table", "sup"]):
        tag.extract()
    return soup.get_text(separator=" ", strip=True)


def embed_on_gpu(gpu_id, docs_chunk):
    print(f"[GPU {gpu_id}] Starting embedding of {len(docs_chunk)} chunks...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": f"cuda:{gpu_id}"}
    )
    return embeddings.embed_documents([d.page_content for d in docs_chunk])

# ---------------- MAIN INGESTION ----------------
if __name__ == "__main__":
    print(f"Starting crawl from {START_URL}")

    headers = {"User-Agent": os.environ["USER_AGENT"]}
    base_html = requests.get(START_URL, headers=headers, timeout=10).text
    links = get_site_links(START_URL, base_html, limit=MAX_LINKS)
    urls = [START_URL] + links[:MAX_LINKS - 1]
    print(f"Found {len(urls)} links to crawl.")

    pages = parallel_fetch(urls, headers, max_workers=MAX_WORKERS)
    print(f"Downloaded {len(pages)} pages successfully.")

    all_documents = [Document(page_content=clean_html(html), metadata={"source": url}) for url, html in pages]

    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        texts = [
            chunk
            for doc_chunks in tqdm(
                executor.map(lambda d: splitter.split_documents([d]), all_documents),
                total=len(all_documents),
                desc="Splitting docs"
            )
            for chunk in doc_chunks
        ]

    # ---------------- MULTI-GPU EMBEDDING ----------------
    print("Creating embeddings across multiple GPUs...")
    free_gpus = get_free_gpus(min_free_mem_gb=35.0)
    num_gpus = len(free_gpus)

    if num_gpus > 1:
        chunks = [texts[i::num_gpus] for i in range(num_gpus)]
        all_embeddings = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = [executor.submit(embed_on_gpu, gpu_id, chunks[i]) for i, gpu_id in enumerate(free_gpus)]
            for f in concurrent.futures.as_completed(futures):
                all_embeddings.extend(f.result())
        embeddings_vectors = all_embeddings
    else:
        print("Only one GPU or CPU detected — using single device.")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": device}
        )
        embeddings_vectors = embeddings.embed_documents([d.page_content for d in texts])

    # ---------------- BUILD FAISS INDEX ----------------
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    import numpy as np
    import faiss, os

    # Assume:
    # texts = [...]  # list of Document objects
    # vectors = [...]  # list of precomputed numpy arrays (one per document)

    # Convert to FAISS index
    embedding_size = len(vectors[0])
    index = faiss.IndexFlatL2(embedding_size)
    index.add(np.array(vectors).astype("float32"))

    # Create embeddings object (so LangChain knows how to embed future queries)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    # Build FAISS store from existing index + docs + embeddings
    db = FAISS(embedding_function=embeddings, index=index, docstore={}, index_to_docstore_id={})

    # Populate the docstore mappings
    for i, doc in enumerate(texts):
        db.docstore[str(i)] = doc
        db.index_to_docstore_id[i] = str(i)

    # Save it
    os.makedirs("faiss_data", exist_ok=True)
    db.save_local("faiss_data/faiss_index")

    print("✅ Saved FAISS index with precomputed embeddings")

    # ---------------- VALIDATION ----------------
    print("\n🔍 Validating FAISS index with a sample query...")
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents("Who is the king of Norway?")
    for i, d in enumerate(docs):
        print(f"\nResult {i+1}:")
        print(d.page_content[:300].replace("\n", " "))
    print("\n🎉 FAISS index check complete — looks good!")
