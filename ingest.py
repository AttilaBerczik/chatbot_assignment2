import os

import embeddings
import requests
import concurrent.futures
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from urllib.parse import urljoin, urlparse

# Model paths (same as in app.py - use pre-downloaded models)
MODELS_DIR = os.environ.get("HF_MODELS_DIR", os.path.join(os.getcwd(), "models"))
EMB_LOCAL_DIR = os.path.join(MODELS_DIR, "bge-large-en-v1.5")

# ---------------- SETTINGS ----------------
START_URL = "https://en.wikipedia.org/wiki/Norway"  # Change this for any site
MAX_LINKS = 10       # Total pages to crawl
MAX_WORKERS = 20       # How many threads to use
os.environ["USER_AGENT"] = "FastCrawlerBot/1.0 (+https://example.com)"

# ---------------- HELPER FUNCTIONS ----------------
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
        print(f"⚠Failed: {url} -> {e}")
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

    # 1️⃣ Fetch base page
    headers = {"User-Agent": os.environ["USER_AGENT"]}
    base_html = requests.get(START_URL, headers=headers, timeout=10).text

    # 2️⃣ Extract internal links (same domain)
    links = get_site_links(START_URL, base_html, limit=MAX_LINKS)
    urls = [START_URL] + links[:MAX_LINKS - 1]
    print(f"🔗 Found {len(urls)} links to crawl.")

    # 3️⃣ Download all pages in parallel
    pages = parallel_fetch(urls, headers, max_workers=MAX_WORKERS)
    print(f"Downloaded {len(pages)} pages successfully.")

    # 4️⃣ Convert to LangChain Documents
    all_documents = [Document(page_content=html, metadata={"source": url}) for url, html in pages]

    # 5️⃣ Split into chunks for embedding
    print("Splitting text into chunks...")
    splitter = SentenceTransformersTokenTextSplitter(chunk_size=512, chunk_overlap=50)


    def split_one(doc):
        return splitter.split_documents([doc])


    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        parts = list(executor.map(split_one, all_documents))

    texts = [chunk for sublist in parts for chunk in sublist]


    print("Convert text into embeddings...")
    # 5️⃣ Convert text into embeddings
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"}
    )

    print("Build and save FAISS index...")
    # 6️⃣ Build and save FAISS index
    db = FAISS.from_documents(texts, embeddings)

    # Save the vector store to a local file
    FAISS_INDEX_PATH = os.path.join("faiss_data", "faiss_index")
    os.makedirs("faiss_data", exist_ok=True)
    db.save_local(FAISS_INDEX_PATH)


    print("Ingestion complete. Vector store saved to ./faiss_index/")
