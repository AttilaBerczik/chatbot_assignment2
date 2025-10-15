import os
import re
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --------------------------------------
# Configuration
# --------------------------------------
START_URL = "https://en.wikipedia.org/wiki/Norway"
MAX_LINKS = 10   # limit for safety
MAX_DEPTH = 1    # how deep to crawl (1 = main page + sublinks)
SAVE_PATH = "faiss_index"

# --------------------------------------
# Helper: Crawl sublinks
# --------------------------------------
def get_sublinks(url, max_links=10, same_domain=True):
    """Extract sublinks from a given page."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"[WARN] Could not load {url}: {e}")
        return []

    base_domain = urlparse(url).netloc
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Skip fragments, mailto, javascript, etc.
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue

        full_url = urljoin(url, href)
        if same_domain and urlparse(full_url).netloc != base_domain:
            continue

        # Basic filtering to avoid non-HTML content
        if re.search(r"\.(jpg|jpeg|png|gif|pdf|zip|mp4|svg)$", full_url, re.IGNORECASE):
            continue

        links.add(full_url)
        if len(links) >= max_links:
            break

    return list(links)

# --------------------------------------
# Recursive crawler
# --------------------------------------
def crawl(url, depth=0, max_depth=1, visited=None, max_links=10):
    if visited is None:
        visited = set()
    if depth > max_depth or url in visited:
        return []

    visited.add(url)
    urls = [url]
    print(f"[CRAWL] Depth {depth}: {url}")

    if depth < max_depth:
        sublinks = get_sublinks(url, max_links=max_links)
        for link in sublinks:
            urls.extend(crawl(link, depth + 1, max_depth, visited, max_links))

    return urls

# --------------------------------------
# Main
# --------------------------------------
if __name__ == "__main__":
    # Step 1: Crawl all URLs (main + sublinks)
    all_urls = crawl(START_URL, depth=0, max_depth=MAX_DEPTH, max_links=MAX_LINKS)
    print(f"\n[Crawl Complete] Total pages collected: {len(all_urls)}")

    # Step 2: Load documents
    print("\n[Load] Fetching documents...")
    loader = WebBaseLoader(all_urls)
    documents = loader.load()

    # Step 3: Split into smaller chunks
    print("[Split] Splitting text into chunks...")
    text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=512, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Step 4: Create embeddings
    print("[Embed] Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 5: Build FAISS index
    print("[FAISS] Building and saving vector store...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(SAVE_PATH)

    print(f"[Done] Vector store created and saved to {SAVE_PATH}.")
