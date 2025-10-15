import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import concurrent.futures
from tqdm import tqdm

# Set a polite User-Agent
os.environ["USER_AGENT"] = "MyLangchainBot/1.0 (+https://example.com)"

def get_internal_links(base_url, limit=10):
    """Extract internal links from any website (same domain only)."""
    try:
        response = requests.get(base_url, headers={"User-Agent": os.environ["USER_AGENT"]}, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to fetch {base_url}: {e}")
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
    all_documents = []
    for url in urls:
        loader = WebBaseLoader(url)
        try:
            docs = loader.load()
            all_documents.extend(docs)
            print(f"✅ Loaded {url}")
        except Exception as e:
            print(f"⚠️ Failed to load {url}: {e}")

    print(f"📄 Loaded {len(all_documents)} total documents.")

    if not all_documents:
        print("❌ No documents loaded. Exiting.")
        return

    # 3️⃣ Split into chunks
    print("Splitting text into chunks...")

    from transformers import AutoTokenizer

    splitter = SentenceTransformersTokenTextSplitter(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=512,
        chunk_overlap=50,
    )

    def split_one(doc):
        return splitter.split_documents([doc])

    # Keep everything inside the with-block
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        texts = [
            chunk
            for doc_chunks in tqdm(
                executor.map(split_one, all_documents),
                total=len(all_documents),
                desc="Splitting docs"
            )
            for chunk in doc_chunks
        ]
    print(f"✂️ Split into {len(texts)} chunks.")

    # 4️⃣ Generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 5️⃣ Build and save FAISS index
    db = FAISS.from_documents(texts, embeddings)

    FAISS_INDEX_PATH = os.path.join("faiss_data", "faiss_index")
    os.makedirs("faiss_data", exist_ok=True)
    db.save_local(FAISS_INDEX_PATH)
    print("💾 Vector store created and saved to faiss_data/faiss_index")


if __name__ == "__main__":
    # Example: crawl any site — not just Wikipedia
    base_url = "https://en.wikipedia.org/wiki/Alder_Dam"  # change this to any site
    crawl_and_embed(base_url, link_limit=10)
