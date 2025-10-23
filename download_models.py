#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

# Unified cache directory for both LLM and embeddings
MODELS_CACHE_DIR = os.environ.get("MODELS_CACHE_DIR", os.path.expanduser("~/chatbot_assignment2/models_cache"))
LLM_REPO = os.environ.get("LLM_REPO", "Qwen/Qwen2.5-7B-Instruct")
EMB_REPO = os.environ.get("EMB_REPO", "BAAI/bge-large-en-v1.5")

# Local layout must match app expectations
LLM_LOCAL_DIR = os.environ.get(
    "LLM_LOCAL_DIR",
    os.path.join(MODELS_CACHE_DIR, "Qwen", "Qwen2.5-7B-Instruct")
)
EMB_LOCAL_DIR = os.environ.get(
    "EMB_LOCAL_DIR",
    os.path.join(MODELS_CACHE_DIR, "bge-large-en-v1.5")
)

os.makedirs(LLM_LOCAL_DIR, exist_ok=True)
os.makedirs(EMB_LOCAL_DIR, exist_ok=True)


def ensure_snapshot(repo_id: str, local_dir: str):
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"[download_models] Found {local_dir}, skipping download for {repo_id}")
        return
    print(f"[download_models] Downloading {repo_id} to {local_dir} ...")
    snapshot_download(repo_id, local_dir=local_dir)


def main():
    ensure_snapshot(LLM_REPO, LLM_LOCAL_DIR)
    ensure_snapshot(EMB_REPO, EMB_LOCAL_DIR)
    print("[download_models] Done.")


if __name__ == "__main__":
    main()
