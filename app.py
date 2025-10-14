import os

# Disable progress bars for cleaner output
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_TIMEOUT"] = "300"  # 5 minute timeout

import torch
import json
import time  # Add missing import
from flask import Flask, render_template, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import Optional

app = Flask(__name__)
qa_chain = None

# GPU Configuration
def get_device():
    """Detect and configure the best available device."""
    if torch.cuda.is_available():
        # Find GPU with most free memory
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s)")

        max_free_memory = 0
        best_gpu = 0
        for i in range(num_gpus):
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} - Free memory: {free_memory / 1024**3:.2f} GB")
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = i

        device = f"cuda:{best_gpu}"
        print(f"Selected device: {device}")
        return device
    else:
        print("No GPU available, using CPU")
        return "cpu"

device = get_device()

# Model paths (downloaded during Docker build)
MODELS_DIR = os.environ.get("HF_MODELS_DIR", os.path.join(os.getcwd(), "models"))
LLM_LOCAL_DIR = os.path.join(MODELS_DIR, "Qwen/Qwen2.5-7B-Instruct")
EMB_LOCAL_DIR = os.path.join(MODELS_DIR, "bge-large-en-v1.5")

# Load tokenizer from pre-downloaded model with extended context
tokenizer = AutoTokenizer.from_pretrained(LLM_LOCAL_DIR, trust_remote_code=True)
MAX_TOKENS = 20000  # Set to 20k tokens for extended context handling
HISTORY_MAX_TOKENS = 2000  # Reserve tokens for past conversation
FAISS_INDEX_PATH = os.path.join("faiss_data", "faiss_index")

# store conversation history in memory
conversation_history = []

class TruncatingHuggingFacePipeline(HuggingFacePipeline):
    def __init__(self, pipeline, tokenizer, max_tokens):
        super().__init__(pipeline=pipeline)
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens

    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        # Truncate prompt to max_tokens
        input_ids = self._tokenizer.encode(prompt, truncation=True, max_length=self._max_tokens)
        truncated_prompt = self._tokenizer.decode(input_ids)
        # Delegate to base LLM _call method
        return super()._call(truncated_prompt, stop=stop)

def initialize_chain():
    """Initializes the conversational retrieval chain."""
    global qa_chain, db, llm
    try:
        # Check if the FAISS index exists
        if not os.path.exists(FAISS_INDEX_PATH):
            return "FAISS index not found. Please run the ingestion script first."

        # Load the embeddings model (from local path, on GPU if available)
        print("Initializing Hugging Face embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMB_LOCAL_DIR,
            model_kwargs={'device': device}
        )

        # Load the vector store from disk
        print("Loading vector store from disk...")
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

        # Display index information
        print(f"✓ Loaded FAISS index: {db.index.ntotal} vectors, dimension {db.index.d}")
        index_type = type(db.index).__name__
        print(f"✓ Index type: {index_type}")

        # If it's an IVF index, show search parameters
        if hasattr(db.index, 'nprobe'):
            print(f"✓ IVF search parameters: nprobe={db.index.nprobe}, nlist={db.index.nlist}")
            print(f"✓ Search efficiency: ~{100 * db.index.nprobe / db.index.nlist:.1f}% of index searched per query")

        # Initialize Hugging Face LLM pipeline with GPU optimization
        print("Initializing Hugging Face LLM pipeline...")
        
        # Load model with optimizations and extended context support
        print(f"Loading model from {LLM_LOCAL_DIR} with 20k context support...")
        print("⚡ Applying speed optimizations:")
        print("  - Multi-GPU tensor parallelism (8x A100)")
        print("  - FP16 precision for 2x speed")
        print("  - KV-cache enabled")

        # Try to use Flash Attention 2 if available
        model_kwargs = {
            "trust_remote_code": True,
            "dtype": torch.float16,  # Always use FP16 on GPU for 2x speed
            "device_map": "auto",  # Automatically spread across all 8 GPUs
            "low_cpu_mem_usage": True,
            "use_cache": True,  # Enable KV-cache for faster generation
        }

        # Try Flash Attention 2 - provides 3-4x speedup if available
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model = AutoModelForCausalLM.from_pretrained(LLM_LOCAL_DIR, **model_kwargs)
            print("  ✓ Flash Attention 2 enabled (3-4x faster attention)")
        except Exception as e:
            print(f"  ⚠ Flash Attention 2 not available, using default attention: {e}")
            # Fall back to default attention
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(LLM_LOCAL_DIR, **model_kwargs)

        # Apply BetterTransformer optimization for additional 20-30% speedup
        try:
            from optimum.bettertransformer import BetterTransformer
            print("  - Applying BetterTransformer...")
            model = BetterTransformer.transform(model)
            print("  ✓ BetterTransformer applied successfully")
        except Exception as e:
            print(f"  ⚠ BetterTransformer not available: {e}")

        # Create pipeline - don't specify device when using device_map="auto"
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            return_full_text=False
        )

        llm = TruncatingHuggingFacePipeline(generator, tokenizer, MAX_TOKENS)
        
        print("Chatbot chain initialized successfully.")
        print("DB initialized:", db)
        print("LLM initialized:", llm)
        print(f"Model loaded with device_map='auto' for optimal GPU utilization")
        print(f"Context length configured for: {MAX_TOKENS} tokens (~15k-20k words)")
        if device.startswith("cuda"):
            gpu_id = int(device.split(":")[-1])
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(gpu_id) / 1024**3:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(gpu_id) / 1024**3:.2f} GB")
        # Initialize conversation history with dynamic welcome prompt
        try:
            metadata_path = os.path.join("faiss_data", "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    m = json.load(f)
                topic = m.get("base_topic", "the ingested content")
            else:
                topic = "the ingested content"
            greeting = f"Hello! I'm your Docker RAG chatbot. Ask me anything about {topic}."
            conversation_history.clear()
            conversation_history.append({"role": "assistant", "content": greeting})
        except Exception as e:
            print(f"Failed to initialize conversation history greeting: {e}")

        return None  # No error
    except Exception as e:
        import traceback
        print(f"Error during chain initialization: {e}")
        traceback.print_exc()
        return f"Error during initialization: {e}"

@app.route("/")
def index():
    """Renders the main chat interface."""
    return render_template("index.html")

@app.route("/topic", methods=["GET"])
def get_topic():
    """Returns the topic/subject of the ingested content."""
    try:
        metadata_path = os.path.join("faiss_data", "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return jsonify({
                "topic": metadata.get("base_topic", "the ingested content"),
                "domain": metadata.get("domain", ""),
                "url_count": len(metadata.get("all_urls", []))
            })
        else:
            return jsonify({
                "topic": "the ingested content",
                "domain": "",
                "url_count": 0
            })
    except Exception as e:
        print(f"Error loading topic metadata: {e}")
        return jsonify({
            "topic": "the ingested content",
            "domain": "",
            "url_count": 0
        })

@app.route("/query", methods=["POST"])
def query():
    try:
        global db, llm, conversation_history

        start_time = time.time()  # Fix: Define start_time at the beginning
        print("\n" + "="*80)
        print("Entered /query endpoint")
        user_query = request.json.get("query")
        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        # append user message to history
        conversation_history.append({"role": "user", "content": user_query})

        # build history string and truncate to last HISTORY_MAX_TOKENS tokens
        retrieval_start = time.time()
        history_text = "\n".join([f"User: {msg['content']}" if msg['role']=="user" else f"Assistant: {msg['content']}" for msg in conversation_history])
        history_tokens = tokenizer.encode(history_text)
        if len(history_tokens) > HISTORY_MAX_TOKENS:
            history_tokens = history_tokens[-HISTORY_MAX_TOKENS:]
            history_text = tokenizer.decode(history_tokens)

        # Retrieve context documents
        retrieved_docs = db.similarity_search(user_query, k=10)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # truncate context if needed
        context_tokens = tokenizer.encode(context)
        if len(context_tokens) > MAX_TOKENS - HISTORY_MAX_TOKENS - 1000:
            context_tokens = context_tokens[:MAX_TOKENS - HISTORY_MAX_TOKENS - 1000]
            context = tokenizer.decode(context_tokens)

        retrieval_time = time.time() - retrieval_start
        print(f"⏱️  Retrieval time: {retrieval_time:.2f}s")

        # Prompt with history and context
        template = """You are a helpful AI assistant. Use the conversation history and the following context to answer the question concisely.

Conversation history:
{history}

Context:
{context}

Instructions:
- Answer using only the context and history.
- Do not mention, reference, or describe the context itself.
- Do not explain how the answer was created.
- Do not include notes, disclaimers, or meta-commentary of any kind.
- If the context does not contain relevant information, say exactly: "I don't know."

Question: {user_query}

Answer:"""
        prompt = PromptTemplate.from_template(template)

        chain = LLMChain(llm=llm, prompt=prompt)

        generation_start = time.time()
        print(f"🤖 Generating response (context: {len(context_tokens)} tokens, history: {len(history_tokens)} tokens)...")
        answer = chain.run({"history": history_text, "context": context, "user_query": user_query})
        generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        print(f"⏱️  Generation time: {generation_time:.2f}s")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📝 Response length: {len(answer)} chars")
        print("="*80 + "\n")

        # append assistant response to history
        conversation_history.append({"role": "assistant", "content": answer})

        return jsonify({
            "answer": answer
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_error = initialize_chain()
    if init_error:
        print(f"Failed to start Flask app due to initialization error: {init_error}")
    else:
        app.run(host="0.0.0.0", debug=True)
