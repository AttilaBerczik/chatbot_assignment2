import os

# Disable progress bars for cleaner output
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_TIMEOUT"] = "300"  # 5 minute timeout

import torch
from flask import Flask, render_template, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

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

# Model configuration
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # Using available Qwen model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
MAX_TOKENS = 4096  # Increased for better context handling
FAISS_INDEX_PATH = os.path.join("faiss_data", "faiss_index")

class TruncatingHuggingFacePipeline(HuggingFacePipeline):
    def __init__(self, pipeline, tokenizer, max_tokens):
        super().__init__(pipeline=pipeline)
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens

    def __call__(self, prompt, stop=None):
        # Handle batch, dict, or string
        if isinstance(prompt, list):
            truncated_list = []
            for item in prompt:
                if isinstance(item, dict):
                    prompt_text = item.get("text") or item.get("inputs") or item.get("prompt") or ""
                else:
                    prompt_text = item
                input_ids = self._tokenizer.encode(prompt_text, truncation=True, max_length=self._max_tokens)
                truncated_prompt = self._tokenizer.decode(input_ids)
                truncated_list.append(truncated_prompt)
            return super().__call__(truncated_list, stop=stop)
        elif isinstance(prompt, dict):
            prompt_text = prompt.get("text") or prompt.get("inputs") or prompt.get("prompt") or ""
            input_ids = self._tokenizer.encode(prompt_text, truncation=True, max_length=self._max_tokens)
            truncated_prompt = self._tokenizer.decode(input_ids)
            return super().__call__(truncated_prompt, stop=stop)
        else:
            input_ids = self._tokenizer.encode(prompt, truncation=True, max_length=self._max_tokens)
            truncated_prompt = self._tokenizer.decode(input_ids)
            return super().__call__(truncated_prompt, stop=stop)

    def _call(self, prompt, stop=None):
        return self.__call__(prompt, stop=stop)

def initialize_chain():
    """Initializes the conversational retrieval chain."""
    global qa_chain, db, llm
    try:
        # Check if the FAISS index exists
        if not os.path.exists(FAISS_INDEX_PATH):
            return "FAISS index not found. Please run the ingestion script first."

        # Load the embeddings model (on GPU if available)
        print("Initializing Hugging Face embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': device}
        )

        # Load the vector store from disk
        print("Loading vector store from disk...")
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

        # Initialize Hugging Face LLM pipeline with GPU optimization
        print("Initializing Hugging Face LLM pipeline...")
        
        # Load model with optimizations
        print(f"Loading model {MODEL_NAME} on {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            device_map=device if device.startswith("cuda") else None,
            low_cpu_mem_usage=True
        )

        # Create pipeline with GPU settings
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            device=device if device.startswith("cuda") else -1,
            return_full_text=False
        )

        llm = TruncatingHuggingFacePipeline(generator, tokenizer, MAX_TOKENS)
        
        print("Chatbot chain initialized successfully.")
        print("DB initialized:", db)
        print("LLM initialized:", llm)
        print(f"Model loaded on: {device}")
        if device.startswith("cuda"):
            gpu_id = int(device.split(":")[-1])
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(gpu_id) / 1024**3:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(gpu_id) / 1024**3:.2f} GB")
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

@app.route("/query", methods=["POST"])
def query():
    try:
        print("Entered /query endpoint")
        global db, llm

        print("Request JSON:", request.json)
        user_query = request.json.get("query")
        print("User query:", user_query)
        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        retrieved_docs = db.similarity_search(user_query, k=3)  # Limit to top 3 for better focus
        for doc in retrieved_docs:
            print(doc.page_content)

        # Build prompt with retrieved context
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Improved prompt template for instruction-following models
        template = """You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {user_query}

Answer:"""

        prompt = PromptTemplate.from_template(template)

        chain = LLMChain(llm=llm, prompt=prompt)
        print("Prompt input:", {"context": context[:200], "user_query": user_query})  # Truncate for logging
        answer = chain.run({"context": context, "user_query": user_query})
        print("Generated answer:", answer)
        if not answer:
            return jsonify({"error": "No answer could be generated."}), 500
        return jsonify({"answer": answer})
    except Exception as e:
        import traceback
        print("Error processing query (outer):")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_error = initialize_chain()
    if init_error:
        print(f"Failed to start Flask app due to initialization error: {init_error}")
    else:
        app.run(host="0.0.0.0", debug=True)
