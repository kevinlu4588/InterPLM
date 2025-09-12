#!/bin/bash

# Script to run the InterPLM dashboard with remote data

# Check if repo_id is provided
if [ "$#" -eq 0 ]; then
    REPO_ID="kevinlu4588/interplm-data"
    echo "Using default repository: $REPO_ID"
else
    REPO_ID="$1"
    echo "Using repository: $REPO_ID"
fi

# Navigate to the dashboard directory
cd interplm/dashboard

echo "Starting InterPLM Dashboard..."
echo "Loading data from Hugging Face: $REPO_ID"
echo ""

# Run with streamlit (the correct way)
streamlit run app_remote.py -- --source remote --repo_id "$REPO_ID"