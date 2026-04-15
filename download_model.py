from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MiniLLM/MiniPLM-Qwen-500M",
    local_dir="./model",
    local_dir_use_symlinks=False
)
