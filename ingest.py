import os
import requests
import json
import faiss
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Model paths (same as in app.py - use pre-downloaded models)
MODELS_DIR = os.environ.get("HF_MODELS_DIR", os.path.join(os.getcwd(), "models"))
EMB_LOCAL_DIR = os.path.join(MODELS_DIR, "bge-large-en-v1.5")

def get_page_title(url):
    """Extract the page title from HTML."""
    try:
        html = requests.get(url, headers={"User-Agent": os.environ.get("USER_AGENT", "Mozilla/5.0")}, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        title = soup.find("title")
        if title:
            return title.get_text().strip()
        # Fallback to h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text().strip()
    except:
        pass
    return None

def get_links_from_page(base_url, limit=5):
    """Extract internal links from any webpage (not just Wikipedia)."""
    try:
        html = requests.get(base_url, headers={"User-Agent": os.environ.get("USER_AGENT", "Mozilla/5.0")}, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")

        base_domain = urlparse(base_url).netloc
        links = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)

            # Only include links from the same domain, skip anchors, javascript, etc.
            if (parsed.netloc == base_domain and
                parsed.scheme in ['http', 'https'] and
                not parsed.path.endswith(('.pdf', '.jpg', '.png', '.gif', '.zip')) and
                '#' not in parsed.path):
                links.add(full_url)
                if len(links) >= limit:
                    break
        return list(links)
    except Exception as e:
        print(f"Error extracting links from {base_url}: {e}")
        return []

def extract_domain_name(url):
    """Extract a clean domain name from URL."""
    parsed = urlparse(url)
    domain = parsed.netloc
    # Remove www. prefix
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain

# 1️⃣ Starting page - YOU CAN CHANGE THIS TO ANY WEBSITE
base_url = "https://en.wikipedia.org/wiki/Rickard_Sarby"

print(f"Starting crawl from: {base_url}")

# Get the title of the main page
main_title = get_page_title(base_url)
if not main_title:
    # Fallback: extract from URL
    if "/wiki/" in base_url:
        main_title = base_url.split("/wiki/")[-1].replace("_", " ")
    else:
        main_title = extract_domain_name(base_url)

# Clean up titles for Wikipedia pages, removing source suffix
domain = extract_domain_name(base_url)
if "wikipedia.org" in domain and " - " in main_title:
    main_title = main_title.split(" - ")[0]

print(f"Main topic: {main_title}")

# 2️⃣ Get related links (crawl a few linked pages from the same domain)
related_links = get_links_from_page(base_url, limit=5)
urls = [base_url] + related_links

print(f"Found {len(urls)} URLs to crawl")

# 3️⃣ Load all pages into LangChain Documents
all_documents = []
for url in urls:
    loader = WebBaseLoader(url)
    try:
        docs = loader.load()
        all_documents.extend(docs)
        print(f"✓ Loaded {url}")
    except Exception as e:
        print(f"✗ Failed to load {url}: {e}")

print(f"\nLoaded {len(all_documents)} total documents.")

# 4️⃣ Split long text into chunks for embedding
text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=512, chunk_overlap=50)
texts = text_splitter.split_documents(all_documents)

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

# Save metadata about ingested content
metadata = {
    "base_url": base_url,
    "base_topic": main_title,
    "domain": extract_domain_name(base_url),
    "all_urls": urls,
    "total_documents": len(all_documents),
    "total_chunks": len(texts),
    "n_vectors": n_vectors,
    "vector_dimension": d,
    "index_type": "IVF-PQ" if n_vectors > 100 else "Flat"
}

metadata_path = os.path.join("faiss_data", "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Vector store created and saved to {FAISS_INDEX_PATH}")
print(f"✓ Metadata saved to {metadata_path}")
print(f"✓ Topic: {metadata['base_topic']}")
print(f"✓ Domain: {metadata['domain']}")
print(f"✓ Index type: {metadata['index_type']}")
print(f"✓ Vectors: {metadata['n_vectors']}")
