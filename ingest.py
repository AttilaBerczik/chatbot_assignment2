import os
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Set a polite User-Agent so Wikipedia doesn’t block our requests
os.environ["USER_AGENT"] = "MyLangchainBot/1.0 (+https://example.com)"

def get_wiki_links(base_url, limit=1000):
    """Extract internal Wikipedia links from a page."""
    html = requests.get(base_url, headers={"User-Agent": os.environ["USER_AGENT"]}).text
    soup = BeautifulSoup(html, "html.parser")

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/wiki/") and ":" not in href:  # Skip non-article links (like Help:, File:)
            links.add("https://en.wikipedia.org" + href)
        if len(links) >= limit:
            break
    return list(links)

# 1️⃣ Starting page
base_url = "https://en.wikipedia.org/wiki/Norway"

# 2️⃣ Get related links (crawl a few linked Wikipedia pages)
related_links = get_wiki_links(base_url, limit=1000)
urls = [base_url] + related_links  # Include the original Norway page

# 3️⃣ Load all pages into LangChain Documents
all_documents = []
for url in urls:
    loader = WebBaseLoader(url)
    try:
        docs = loader.load()
        all_documents.extend(docs)
        print(f"Loaded {url}")
    except Exception as e:
        print(f"Failed to load {url}: {e}")

print(f"Loaded {len(all_documents)} total documents.")

# 4️⃣ Split long text into chunks for embedding
text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=512, chunk_overlap=50)
texts = text_splitter.split_documents(all_documents)

# 5️⃣ Convert text into embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 6️⃣ Build and save FAISS index
db = FAISS.from_documents(texts, embeddings)

# Save the vector store to a local file
FAISS_INDEX_PATH = os.path.join("faiss_data", "faiss_index")
os.makedirs("faiss_data", exist_ok=True)
db.save_local(FAISS_INDEX_PATH)

print("✅ Vector store created and saved to faiss_index.")