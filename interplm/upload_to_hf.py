"""
Script to upload InterPLM models, dashboard cache, and data to Hugging Face Hub.
This allows running the dashboard on any machine without training SAEs locally.
"""

import os
import pickle
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from tap import Tap
from dotenv import load_dotenv
load_dotenv()  # will look for .env in current directory

# Then this works automatically:
token = os.environ.get("HF_TOKEN")
print(token)
class UploadArgs(Tap):
    repo_id: str  # Hugging Face repository ID (e.g., "username/interplm-dashboard-data")
    local_data_dir: str = "."  # Base directory containing all data
    models_dir: Optional[str] = "models"  # Directory containing trained SAE models
    dashboard_cache_path: Optional[str] = "data/dashboard_cache/dashboard_cache.pkl"
    metadata_path: Optional[str] = "data/dashboard_cache/swiss-prot_metadata.tsv.gz"
    token: Optional[str] = None  # Hugging Face token (can use HF_TOKEN env var instead)
    private: bool = False  # Whether to make the repository private
    upload_embeddings: bool = False  # Whether to upload embedding files (large)


def upload_to_huggingface(args: UploadArgs):
    """Upload InterPLM data to Hugging Face Hub."""
    
    # Initialize HF API
    api = HfApi(token=args.token or os.environ.get("HF_TOKEN"))
    
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
            token=args.token
        )
        print(f"Repository {args.repo_id} created/verified")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    base_path = Path(args.local_data_dir)
    
    # Upload dashboard cache
    if args.dashboard_cache_path:
        cache_path = base_path / args.dashboard_cache_path
        if cache_path.exists():
            print(f"Uploading dashboard cache from {cache_path}...")
            try:
                upload_file(
                    path_or_fileobj=str(cache_path),
                    path_in_repo="dashboard_cache/dashboard_cache.pkl",
                    repo_id=args.repo_id,
                    repo_type="model",
                    token=args.token
                )
                print("✓ Dashboard cache uploaded")
            except Exception as e:
                print(f"Error uploading dashboard cache: {e}")
    
    # Upload metadata
    if args.metadata_path:
        metadata_path = base_path / args.metadata_path
        if metadata_path.exists():
            print(f"Uploading metadata from {metadata_path}...")
            try:
                upload_file(
                    path_or_fileobj=str(metadata_path),
                    path_in_repo="dashboard_cache/swiss-prot_metadata.tsv.gz",
                    repo_id=args.repo_id,
                    repo_type="model",
                    token=args.token
                )
                print("✓ Metadata uploaded")
            except Exception as e:
                print(f"Error uploading metadata: {e}")
    
    # Upload models directory
    if args.models_dir:
        models_path = base_path / args.models_dir
        if models_path.exists():
            print(f"Uploading models from {models_path}...")
            try:
                # Upload each model directory separately for better organization
                for model_dir in models_path.iterdir():
                    if model_dir.is_dir():
                        print(f"  Uploading {model_dir.name}...")
                        upload_folder(
                            folder_path=str(model_dir),
                            path_in_repo=f"models/{model_dir.name}",
                            repo_id=args.repo_id,
                            repo_type="model",
                            token=args.token,
                            ignore_patterns=["*.pyc", "__pycache__", ".git"]
                        )
                print("✓ Models uploaded")
            except Exception as e:
                print(f"Error uploading models: {e}")
    
    # Optionally upload embeddings (these can be very large)
    if args.upload_embeddings:
        embeddings_dirs = [
            "data/esm_embds",
            "data/uniprotkb/embeddings"
        ]
        for emb_dir in embeddings_dirs:
            emb_path = base_path / emb_dir
            if emb_path.exists():
                print(f"Uploading embeddings from {emb_path}...")
                try:
                    upload_folder(
                        folder_path=str(emb_path),
                        path_in_repo=emb_dir,
                        repo_id=args.repo_id,
                        repo_type="model",
                        token=args.token,
                        ignore_patterns=["*.pyc", "__pycache__"]
                    )
                    print(f"✓ Embeddings from {emb_dir} uploaded")
                except Exception as e:
                    print(f"Error uploading embeddings: {e}")
    
    # Upload concept analysis results if they exist
    results_path = base_path / "results"
    if results_path.exists():
        print(f"Uploading results from {results_path}...")
        try:
            upload_folder(
                folder_path=str(results_path),
                path_in_repo="results",
                repo_id=args.repo_id,
                repo_type="model",
                token=args.token,
                ignore_patterns=["*.pyc", "__pycache__"]
            )
            print("✓ Results uploaded")
        except Exception as e:
            print(f"Error uploading results: {e}")
    
    print(f"\n✅ Upload complete! Your data is now available at:")
    print(f"   https://huggingface.co/{args.repo_id}")
    print(f"\nTo use this data with the dashboard, run:")
    print(f"   python interplm/dashboard/app_remote.py --repo_id {args.repo_id}")


if __name__ == "__main__":
    args = UploadArgs().parse_args()
    upload_to_huggingface(args)