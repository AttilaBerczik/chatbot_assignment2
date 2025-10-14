import os
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
from huggingface_hub import snapshot_download

# ------------------ CACHE SETUP ------------------
CACHE_DIR = os.path.join(os.getcwd(), "models_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Make Hugging Face + Transformers + SentenceTransformers use it
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

MODEL_NAME = "BAAI/bge-large-en-v1.5"
#print(f"Hugging Face cache dir: {os.environ['HF_HOME']}")

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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
import subprocess
import re
from tqdm import tqdm
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS

# ---------------- GPU DETECTION ----------------
def get_free_gpus(min_free_mem_gb=35.0):
    """
    Returns a list of GPU IDs that have more than `min_free_mem_gb` GB free memory.
    """
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
        return list(range(torch.cuda.device_count()))  # fallback: use all GPUs


def get_device():
    """Detect and configure the best available device (GPU if possible)."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        #print(f"Using GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
        max_free_memory = 0
        best_gpu = 0
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            free_memory = props.total_memory - torch.cuda.memory_allocated(i)
            #print(f"GPU {i}: {props.name} | Free memory: {free_memory / 1024**3:.2f} GB")
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = i
        device = f"cuda:{best_gpu}"
        #print(f"Using GPU device: {device}")
        return device
    else:
        print("No GPU available — using CPU instead.")
        return "cpu"

device = get_device()

# ---------------- SETTINGS ----------------
MODELS_DIR = os.environ.get("HF_MODELS_DIR", os.path.join(os.getcwd(), "models"))
EMB_LOCAL_DIR = os.path.join(MODELS_DIR, "bge-large-en-v1.5")
START_URL = "https://en.wikipedia.org/wiki/Norway"  # Change this for any site
MAX_LINKS = 5
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

def clean_html(raw_html):
    """Strip tags, scripts, and extract readable text."""
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text)
    return text

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

    print("Splitting text into chunks. ..")
    #splitter = SentenceTransformersTokenTextSplitter(chunk_size=512, chunk_overlap=50)
    print("Splitting text into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )


    def split_one(doc):
        return splitter.split_documents([doc])


    # Keep everything inside the with-block
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        texts = [
            chunk
            for doc_chunks in tqdm(
                executor.map(split_one, all_documents),
                total=len(all_documents),
                desc="Splitting docs"
            )
            for chunk in doc_chunks
        ]


    #def split_one(doc):
    #   return splitter.split_documents([doc])
    #
    #with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    #    parts = list(executor.map(split_one, all_documents))

    #texts = [chunk for sublist in parts for chunk in sublist]


    # ---------------- BUILD FAISS INDEX ----------------
    print("Building FAISS vector store...")

    # Convert Document objects to plain strings for FAISS
    text_contents = [d.page_content for d in texts]


    # Reuse same model and ensure GPU
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda"}
    )

    os.makedirs("faiss_data", exist_ok=True)
    db = FAISS.from_texts(text_contents, embeddings)
    db.save_local("faiss_data/faiss_index")

    print("Ingestion complete! Vector store saved to faiss_data/faiss_index")