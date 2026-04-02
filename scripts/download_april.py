import os
import zipfile
import json
from huggingface_hub import hf_hub_download
from pathlib import Path
from loguru import logger

def main():
    data_dir = Path("data/april")
    data_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "uw-math-ai/APRIL"
    filename = "data_by_split.zip"

    logger.info(f"Downloading {filename} from Hugging Face ({repo_id})...")
    try:
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=data_dir
        )
        
        logger.info(f"Extracting {zip_path} to {data_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract everything
            zip_ref.extractall(data_dir)
            
        # APRIL usually extracts into subfolders or with specific names.
        # Let's flatten/rename them to train.jsonl and val.jsonl for our DataModule.
        
        # Look for the .jsonl files in the extracted content
        jsonl_files = list(data_dir.rglob("*.jsonl"))
        logger.info(f"Extracted files: {[f.name for f in jsonl_files]}")
        
        for f in jsonl_files:
            if "train" in f.name.lower():
                os.rename(f, data_dir / "train.jsonl")
            elif "val" in f.name.lower() or "validation" in f.name.lower():
                os.rename(f, data_dir / "val.jsonl")
            elif "test" in f.name.lower():
                os.rename(f, data_dir / "test.jsonl")

        # Clean up the zip file
        if os.path.exists(zip_path) and str(zip_path).endswith(".zip"):
             os.remove(zip_path)

        logger.info("APRIL dataset downloaded, extracted, and renamed successfully.")
        
    except Exception as e:
        logger.error(f"Failed to download/extract APRIL dataset: {e}")

if __name__ == "__main__":
    main()
