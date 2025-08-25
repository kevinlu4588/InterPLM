"""
Data loader utilities for loading InterPLM data from local or remote sources.
Supports loading from Hugging Face Hub with automatic caching.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from interplm.constants import DASHBOARD_CACHE_DIR
from interplm.utils import get_device


class DataLoader:
    """Handles loading data from local filesystem or Hugging Face Hub."""
    
    def __init__(
        self,
        source: str = "local",
        repo_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None
    ):
        """
        Initialize data loader.
        
        Args:
            source: "local" or "remote" (Hugging Face)
            repo_id: Hugging Face repository ID (required if source="remote")
            cache_dir: Directory to cache downloaded files
            token: Hugging Face token for private repos
        """
        self.source = source
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        
        # Set cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "interplm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if source == "remote" and not repo_id:
            raise ValueError("repo_id must be provided when source='remote'")
    
    def load_dashboard_cache(self) -> Dict:
        """Load the main dashboard cache file."""
        if self.source == "local":
            cache_path = DASHBOARD_CACHE_DIR / "dashboard_cache.pkl"
            if not cache_path.exists():
                raise FileNotFoundError(f"Dashboard cache not found at {cache_path}")
            
            print(f"Loading local dashboard cache from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        else:  # remote
            print(f"Downloading dashboard cache from {self.repo_id}...")
            cache_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="dashboard_cache/dashboard_cache.pkl",
                cache_dir=self.cache_dir,
                token=self.token,
                local_dir=self.cache_dir / "dashboard_cache"
            )
            
            print(f"Loading dashboard cache from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    
    def load_protein_metadata(self) -> pd.DataFrame:
        """Load Swiss-Prot protein metadata."""
        if self.source == "local":
            metadata_path = DASHBOARD_CACHE_DIR / "swiss-prot_metadata.tsv.gz"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found at {metadata_path}")
            
            print(f"Loading local metadata from {metadata_path}")
            return pd.read_csv(metadata_path, sep="\t", index_col=0)
        
        else:  # remote
            print(f"Downloading metadata from {self.repo_id}...")
            metadata_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="dashboard_cache/swiss-prot_metadata.tsv.gz",
                cache_dir=self.cache_dir,
                token=self.token,
                local_dir=self.cache_dir / "dashboard_cache"
            )
            
            print(f"Loading metadata from {metadata_path}")
            return pd.read_csv(metadata_path, sep="\t", index_col=0)
    
    def load_model(self, model_name: str, layer: int) -> Optional[torch.nn.Module]:
        """
        Load a specific SAE model.
        
        Args:
            model_name: Name of the model directory
            layer: Layer number
        
        Returns:
            Loaded model or None if not found
        """
        if self.source == "local":
            model_path = Path("models") / model_name / "ae_normalized.pt"
            if not model_path.exists():
                model_path = Path("models") / model_name / "ae.pt"
            
            if not model_path.exists():
                print(f"Model not found at {model_path}")
                return None
            
            print(f"Loading local model from {model_path}")
            from interplm.sae.dictionary import AutoEncoder
            return AutoEncoder.from_pretrained(model_path, device=get_device())
        
        else:  # remote
            print(f"Downloading model {model_name} from {self.repo_id}...")
            try:
                # Try normalized version first
                model_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=f"models/{model_name}/ae_normalized.pt",
                    cache_dir=self.cache_dir,
                    token=self.token,
                    local_dir=self.cache_dir / "models" / model_name
                )
            except:
                # Fall back to regular version
                try:
                    model_path = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=f"models/{model_name}/ae.pt",
                        cache_dir=self.cache_dir,
                        token=self.token,
                        local_dir=self.cache_dir / "models" / model_name
                    )
                except:
                    print(f"Model {model_name} not found in repository")
                    return None
            
            print(f"Loading model from {model_path}")
            from interplm.sae.dictionary import AutoEncoder
            return AutoEncoder.from_pretrained(model_path, device=get_device())
    
    def download_all_data(self) -> str:
        """
        Download all data from Hugging Face repository.
        Useful for offline usage after initial download.
        
        Returns:
            Path to downloaded data
        """
        if self.source == "local":
            print("Already using local data")
            return str(DASHBOARD_CACHE_DIR.parent)
        
        print(f"Downloading all data from {self.repo_id}...")
        local_dir = self.cache_dir / "full_download"
        
        snapshot_download(
            repo_id=self.repo_id,
            cache_dir=self.cache_dir,
            token=self.token,
            local_dir=local_dir
        )
        
        print(f"All data downloaded to {local_dir}")
        return str(local_dir)


def get_data_loader(config: Optional[Dict] = None) -> DataLoader:
    """
    Get a data loader based on configuration.
    
    Args:
        config: Configuration dictionary with keys:
            - source: "local" or "remote"
            - repo_id: Hugging Face repository ID
            - cache_dir: Cache directory for downloads
            - token: HF token for private repos
    
    Returns:
        Configured DataLoader instance
    """
    if config is None:
        # Default to local data
        return DataLoader(source="local")
    
    return DataLoader(
        source=config.get("source", "local"),
        repo_id=config.get("repo_id"),
        cache_dir=config.get("cache_dir"),
        token=config.get("token")
    )