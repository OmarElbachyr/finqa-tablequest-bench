#!/usr/bin/env bash
set -e

# Small models (≤5 B parameters)
SMALL_MODELS=(
  "phi3:mini"
  "llama3.2:3b"              
  "deepseek-r1:1.5b"         
  "gemma3:4b"
  "qwen3:4b"
)

# Medium models (∼7–12 B parameters)
MEDIUM_MODELS=(
  "mistral:7b"                  
  "mistral-nemo:12b"
  "gemma3:12b"
  "qwen3:4b"
  "qwen3:14b"
  "deepseek-r1:7b"
  "deepseek-r1:14b"
)

# Large models (≥24 B parameters, quantized)
LARGE_MODELS=(
  "gemma3:27b"        
  "llama3.1:70b"      
  "qwen3:30b"
  "deepseek-r1:32b"
)

ALL_MODELS=( "${SMALL_MODELS[@]}" "${MEDIUM_MODELS[@]}" "${LARGE_MODELS[@]}" )

echo "Pulling ${#ALL_MODELS[@]} Ollama models..."
for MODEL in "${ALL_MODELS[@]}"; do
  echo "→ ollama pull $MODEL"
  ollama pull "$MODEL"
done

echo "All models pulled successfully."

