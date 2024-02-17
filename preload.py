import huggingface_hub
MODEL_REPO = "fancyfeast/joytag"
print("Downloading model...")
path = huggingface_hub.snapshot_download(MODEL_REPO)