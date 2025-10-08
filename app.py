from itertools import chain
import os
import torch
from flask import Flask, render_template, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = Flask(__name__)
qa_chain = None
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
MAX_TOKENS = 1024

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
        if not os.path.exists("faiss_index"):
            return "FAISS index not found. Please run the ingestion script first."

        # Load the embeddings model
        print("Initializing Hugging Face embeddings model...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load the vector store from disk
        print("Loading vector store from disk...")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Initialize Hugging Face LLM pipeline
        print("Initializing Hugging Face LLM pipeline...")
        
        # Load local text generation model
        generator = pipeline("text2text-generation", model="google/flan-t5-base")
        llm = TruncatingHuggingFacePipeline(generator, tokenizer, MAX_TOKENS)
        
        print("Chatbot chain initialized successfully.")
        print("DB initialized:", db)
        print("LLM initialized:", llm)
        return None  # No error
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
            print(doc.page_content)

        # Build prompt with retrieved context
        context = "\n".join([doc.page_content for doc in retrieved_docs])
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
