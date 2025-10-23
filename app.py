from itertools import chain
import os
import torch
from flask import Flask, render_template, request, jsonify

# LangChain (modern structure)
from langchain_community.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

# Hugging Face / Transformers
from transformers import pipeline, AutoTokenizer
app = Flask(__name__)
qa_chain = None
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
MAX_TOKENS = 1024
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
        # Check if FAISS index exists
        if not os.path.exists(FAISS_INDEX_PATH):
            return "FAISS index not found. Please run the ingestion script first."

        # ---- Cache setup ----
        CACHE_DIR = os.path.expanduser("~/chatbot_assignment2/models_cache")
        os.environ["HF_HOME"] = CACHE_DIR  # ensure transformers uses this cache

        # ---- Embeddings model ----
        MODEL_NAME = "BAAI/bge-large-en-v1.5"
        print("Initializing Hugging Face embeddings model...")
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, cache_folder=CACHE_DIR)

        # ---- Vector store ----
        print("Loading vector store from disk...")
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

        # ---- Language model ----
        print("Initializing Hugging Face LLM pipeline...")
        llm_model_name = "google/flan-t5-base"

        generator = pipeline(
            "text2text-generation",
            model=llm_model_name,
            model_kwargs={"torch_dtype": torch.float16},
            device=0 if torch.cuda.is_available() else -1,
            cache_dir=CACHE_DIR,  # <--- store and reuse model here
        )

        tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cache_dir=CACHE_DIR)

        llm = TruncatingHuggingFacePipeline(generator, tokenizer, MAX_TOKENS)

        print("✅ Chatbot chain initialized successfully.")
        return None
    except Exception as e:
        print(f"Error during chain initialization: {e}")
        return f"Error during initialization: {e}"

@app.route("/")
def index():
    """Renders the main chat interface."""
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    try:
        print("Entered /query endpoint")
        global db, llm  # Remove qa_chain

        print("Request JSON:", request.json)
        user_query = request.json.get("query")
        print("User query:", user_query)
        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        retrieved_docs = db.similarity_search(user_query)
        for doc in retrieved_docs:
            print(doc.page_content[:100] + "...")  # Print first 100 chars

        # Build prompt with retrieved context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Check context size and truncate if needed
        context_tokens = tokenizer.encode(context)
        if len(context_tokens) > MAX_TOKENS - 1000:  # Reserve 1000 tokens for question and answer
            print(f"Context too long ({len(context_tokens)} tokens), truncating...")
            context_tokens = context_tokens[:MAX_TOKENS - 1000]
            context = tokenizer.decode(context_tokens)

        # Token-limit safeguard before sending to model
        input_ids = tokenizer.encode(context, add_special_tokens=False)
        if len(input_ids) > MAX_TOKENS:
            print(f"⚠️ Context too long ({len(input_ids)} tokens), truncating to {MAX_TOKENS}")
            input_ids = input_ids[:MAX_TOKENS]
            context = tokenizer.decode(input_ids, skip_special_tokens=True)

        template = "Context:\n{context}\n\nAnswer the question: {user_query}"
        prompt = PromptTemplate.from_template(template)

        chain = LLMChain(llm=llm, prompt=prompt)
        print("Prompt input:", {"context": context, "user_query": user_query})
        answer = chain.run({"context": context, "user_query": user_query})
        print(answer)
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