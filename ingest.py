import os
from huggingface_hub import snapshot_download

# ------------------ CACHE SETUP ------------------

CACHE_DIR = os.path.join(os.getcwd(), "models_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Make Hugging Face + Transformers + SentenceTransformers use it
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

MODEL_NAME = "BAAI/bge-large-en-v1.5"

os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/root/.cache/huggingface"
print(f"Hugging Face cache dir: {os.environ['HF_HOME']}")

# ------------------ DOWNLOAD IF NEEDED ------------------
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

import requests
import concurrent.futures
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_core.documents import Document
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch

# ---------------- GPU DETECTION ----------------
def get_device():
    """Detect and configure the best available device (GPU if possible)."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s)")
        max_free_memory = 0
        best_gpu = 0
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            free_memory = props.total_memory - torch.cuda.memory_allocated(i)
            print(f"GPU {i}: {props.name} | Free memory: {free_memory / 1024**3:.2f} GB")
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = i
        device = f"cuda:{best_gpu}"
        print(f"Using GPU device: {device}")
        return device
    else:
        print("No GPU available — using CPU instead.")
        return "cpu"

device = get_device()

# ---------------- SETTINGS ----------------
MODELS_DIR = os.environ.get("HF_MODELS_DIR", os.path.join(os.getcwd(), "models"))
EMB_LOCAL_DIR = os.path.join(MODELS_DIR, "bge-large-en-v1.5")
START_URL = "https://en.wikipedia.org/wiki/Norway"  # Change this for any site
MAX_LINKS = 1
MAX_WORKERS = 20
os.environ["USER_AGENT"] = "FastCrawlerBot/1.0 (+https://example.com)"

# ---------------- HELPERS ----------------
def get_site_links(base_url, html, limit=1000):
    """Extract same-domain links from a page."""
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
    """Fetch a URL and return (url, html) or (url, None) if failed."""
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        print(f"Loaded: {url}")
        return url, resp.text
    except Exception as e:
        print(f"Failed: {url} -> {e}")
        return url, None

def parallel_fetch(urls, headers, max_workers=10):
    """Download multiple pages in parallel."""
    pages = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch, url, headers): url for url in urls}
        for future in concurrent.futures.as_completed(futures):
            url, content = future.result()
            if content:
                pages.append((url, content))
    return pages

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

    all_documents = [Document(page_content=html, metadata={"source": url}) for url, html in pages]

    print("Splitting text into chunks. ..")
    splitter = SentenceTransformersTokenTextSplitter(chunk_size=512, chunk_overlap=50)


    def split_one(doc):
        return splitter.split_documents([doc])


    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        parts = list(executor.map(split_one, all_documents))

    texts = [chunk for sublist in parts for chunk in sublist]

    #def split_one(doc):
    #   return splitter.split_documents([doc])
    #
    #with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    #    parts = list(executor.map(split_one, all_documents))

    #texts = [chunk for sublist in parts for chunk in sublist]

    # ---------------- MULTI-GPU EMBEDDING ----------------
    import torch
    import concurrent.futures
    from langchain_huggingface import HuggingFaceEmbeddings

    print("Creating embeddings across multiple GPUs...")

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s).")


    def embed_on_gpu(gpu_id, docs_chunk):
        print(f"[GPU {gpu_id}] Starting embedding of {len(docs_chunk)} chunks...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": f"cuda:{gpu_id}"}
        )
        return embeddings.embed_documents([d.page_content for d in docs_chunk])


    # Split work roughly evenly among GPUs
    if num_gpus > 1:
        chunks = [texts[i::num_gpus] for i in range(num_gpus)]

        all_embeddings = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = [executor.submit(embed_on_gpu, i, chunks[i]) for i in range(num_gpus)]
            for f in concurrent.futures.as_completed(futures):
                all_embeddings.extend(f.result())

        # Merge results (you’ll store them next in FAISS)
        embeddings_vectors = all_embeddings
    else:
        print("Only one GPU detected — using cuda:0")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cuda:0"}
        )
        embeddings_vectors = embeddings.embed_documents([d.page_content for d in texts])

    # ---------------- BUILD FAISS INDEX ----------------
    print("Building FAISS vector store...")
    print(f"Embeddings type: {type(embeddings)}")

    db = FAISS.from_embeddings(embeddings_vectors, texts)
    FAISS_INDEX_PATH = os.path.join("faiss_data", "faiss_index")
    os.makedirs("faiss_data", exist_ok=True)
    db.save_local(FAISS_INDEX_PATH)

    print(f"Ingestion complete. Vector store saved to: {FAISS_INDEX_PATH}")