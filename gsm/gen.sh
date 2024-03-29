export AGENTS=3
export ROUNDS=2
export MODEL_ID="meta-llama/Llama-2-7b-chat-hf"
export TOKEN=""
export SPLIT="[/INST]"
export ROLE="assistant"
export OUTPUT_DIR="output"
export CUDA_VISIBLE_DEVICES=1

python gen_gsm.py \
    --agents $AGENTS \
    --rounds $ROUNDS \
    --model_id "$MODEL_ID" \
    --split "$SPLIT" \
    --role "$ROLE" \
    --output_dir "$OUTPUT_DIR" \
    --sys \
    --summarize \
    # --token "$TOKEN" \
    # --api \
