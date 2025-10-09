#!/usr/bin/env python3
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings

# Preload Qwen model
AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True)
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True)

# Preload BGE embeddings and run a dummy embed
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", model_kwargs={'device': 'cpu'})
_ = emb.embed_query("test")

