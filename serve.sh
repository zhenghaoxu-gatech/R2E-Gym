#!/bin/bash

# vLLM server startup script
# Usage: ./serve.sh <model_path_local>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <model_path_local>"
    exit 1
fi

MODEL_PATH_LOCAL="$1"

echo "Starting vLLM server for model: $MODEL_PATH_LOCAL"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

uv run --with 'openai==1.99.9' \
  vllm serve "$MODEL_PATH_LOCAL" \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 8 \
  --max-model-len 65536 \
  --hf-overrides '{"max_position_embeddings": 65536}' \
  --enable_prefix_caching &

echo "vLLM server started in background"
echo "Waiting 300 seconds for server to initialize..."
sleep 300
echo "Server should be ready at http://0.0.0.0:8000"