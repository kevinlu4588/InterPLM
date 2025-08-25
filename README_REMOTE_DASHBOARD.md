# Running InterPLM Dashboard with Remote Data

This guide explains how to upload your trained SAE models and dashboard data to Hugging Face, then run the dashboard on your personal laptop without needing to train the models locally.

## Step 1: Upload Data to Hugging Face (On your training machine)

First, make sure you have a Hugging Face account and have created an access token:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with write access
3. Set it as an environment variable: `export HF_TOKEN=your_token_here`

### Upload your data to Hugging Face:

```bash
# From your training machine where the models and data exist
cd /path/to/InterPLM

# Upload all dashboard data to Hugging Face
python interplm/upload_to_hf.py \
    --repo_id "your-username/interplm-dashboard-data" \
    --local_data_dir "." \
    --models_dir "models" \
    --dashboard_cache_path "data/dashboard_cache/dashboard_cache.pkl" \
    --metadata_path "data/dashboard_cache/swiss-prot_metadata.tsv.gz" \
    --private false  # Set to true for private repository
```

Optional: If you want to include embeddings (these are large):
```bash
python interplm/upload_to_hf.py \
    --repo_id "your-username/interplm-dashboard-data" \
    --local_data_dir "." \
    --upload_embeddings true
```

## Step 2: Set Up Dashboard on Your Local Machine

### Install InterPLM on your laptop:

```bash
# Clone the repository
git clone https://github.com/ElanaPearl/interPLM.git
cd interPLM

# Create and activate conda environment
conda env create -f env.yml
conda activate interplm

# Install package
pip install -e .
```

### Set up authentication (if using private repository):

```bash
export HF_TOKEN=your_token_here
```

## Step 3: Run the Dashboard with Remote Data

### Option A: Command Line Arguments

```bash
cd interplm/dashboard
streamlit run app_remote.py -- \
    --source remote \
    --repo_id "your-username/interplm-dashboard-data"
```

### Option B: Environment Variables

```bash
export INTERPLM_SOURCE=remote
export INTERPLM_REPO_ID="your-username/interplm-dashboard-data"
export HF_TOKEN=your_token_here  # If private repo

cd interplm/dashboard
streamlit run app_remote.py
```

### Option C: Download All Data First (for offline use)

```bash
# This downloads all data to local cache for offline use
cd interplm/dashboard
streamlit run app_remote.py -- \
    --source remote \
    --repo_id "your-username/interplm-dashboard-data" \
    --download_all true
```

## Step 4: Access the Dashboard

After running the command, open your browser and go to:
```
http://localhost:8501
```

## Caching

The dashboard automatically caches downloaded files in:
- Default: `~/.cache/interplm/`
- Custom: Use `--cache_dir /your/path` to specify a different location

Once files are cached, subsequent launches will be much faster.

## Switching Between Local and Remote Data

You can easily switch between local and remote data sources:

```bash
# Use local data (if you have it)
streamlit run app_remote.py -- --source local

# Use remote data
streamlit run app_remote.py -- --source remote --repo_id "your-username/interplm-dashboard-data"
```

## Troubleshooting

1. **Authentication errors**: Make sure your HF_TOKEN is set correctly
2. **Download failures**: Check your internet connection and repository ID
3. **Missing data**: Ensure all required files were uploaded to Hugging Face
4. **Performance issues**: Consider using `--download_all` for better performance

## Data Structure on Hugging Face

Your Hugging Face repository should have this structure:
```
your-username/interplm-dashboard-data/
├── dashboard_cache/
│   ├── dashboard_cache.pkl
│   └── swiss-prot_metadata.tsv.gz
├── models/
│   └── walkthrough_model/
│       ├── ae.pt
│       ├── ae_normalized.pt
│       └── config.json
└── results/
    └── test_counts/
        └── heldout_all_top_pairings.csv
```

## Python API Usage

You can also use the data loader programmatically:

```python
from interplm.dashboard.data_loader import DataLoader

# Load from Hugging Face
loader = DataLoader(
    source="remote",
    repo_id="your-username/interplm-dashboard-data",
    cache_dir="~/.cache/interplm"
)

# Load dashboard data
dash_data = loader.load_dashboard_cache()
protein_metadata = loader.load_protein_metadata()

# Load specific model
model = loader.load_model("walkthrough_model", layer=3)
```