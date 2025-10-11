#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

# Config via environment variables with defaults
HF_MODELS_DIR = os.environ.get("HF_MODELS_DIR", "/opt/models")
LLM_REPO = os.environ.get("LLM_REPO", "Qwen/Qwen2.5-7B-Instruct")
LLM_LOCAL_DIR = os.environ.get("LLM_LOCAL_DIR", os.path.join(HF_MODELS_DIR, "Qwen/Qwen2.5-7B-Instruct"))
EMB_REPO = os.environ.get("EMB_REPO", "BAAI/bge-large-en-v1.5")
EMB_LOCAL_DIR = os.environ.get("EMB_LOCAL_DIR", os.path.join(HF_MODELS_DIR, "bge-large-en-v1.5"))

os.makedirs(HF_MODELS_DIR, exist_ok=True)


def ensure_snapshot(repo_id: str, local_dir: str):
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"[download_models] Found {local_dir}, skipping download for {repo_id}")
        return
    print(f"[download_models] Downloading {repo_id} to {local_dir} ...")
    snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False)


def main():
    ensure_snapshot(LLM_REPO, LLM_LOCAL_DIR)
    ensure_snapshot(EMB_REPO, EMB_LOCAL_DIR)
    print("[download_models] Done.")


if __name__ == "__main__":
    main()

