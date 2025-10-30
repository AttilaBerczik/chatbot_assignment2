import concurrent.futures
import os
import json
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from tqdm import tqdm

# Set a polite User-Agent
os.environ["USER_AGENT"] = "MyLangchainBot/1.0 (+https://example.com)"

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

    # Use the shared models cache directory
    CACHE_DIR = "/root/chatbot_assignment2/models_cache"
    MODEL_NAME = os.path.join(CACHE_DIR, "BAAI__bge-large-en-v1.5")

    print("Splitting text into chunks...")

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
        #model_kwargs={"device": "cuda"},
    )

    # 5️⃣ Build and save FAISS index
    print("Building FAISS index from documents...")
    db = FAISS.from_documents(safe_texts, embeddings)

    # Save the vector store to a local file
    FAISS_INDEX_PATH = os.path.join("faiss_data", "faiss_index")
    os.makedirs("faiss_data", exist_ok=True)
    db.save_local(FAISS_INDEX_PATH)

    # 6️⃣ Save metadata about the ingested content
    parsed_url = urlparse(base_url)
    domain = parsed_url.netloc
    # Extract topic from URL path (last segment)
    path_segments = [s for s in parsed_url.path.split('/') if s]
    topic = path_segments[-1].replace('_', ' ').replace('-', ' ') if path_segments else domain

    metadata = {
        "base_url": base_url,
        "base_topic": topic,
        "domain": domain,
        "all_urls": urls,
        "total_documents": len(safe_texts),
        "ingestion_timestamp": str(time.time())
    }

    metadata_path = os.path.join("faiss_data", "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved. Topic: {topic}")
    print("Vector store created and saved to faiss_data/faiss_index")


if __name__ == "__main__":
    # Example: crawl any site — not just Wikipedia
    base_url = "https://en.wikipedia.org/wiki/Norway"  # change this to any site
    crawl_and_embed(base_url, link_limit=10)
