export AGENTS=4
export ROUNDS=2
export MODEL_ID="meta-llama/Llama-2-7b-chat-hf"
export TOKEN=""
export SPLIT="[/INST]" #"model\n"
export ROLE="assistant" #"model"
export CUDA_VISIBLE_DEVICES=1

python gen_math.py \
    --agents $AGENTS \
    --rounds $ROUNDS \
    --model_id "$MODEL_ID" \
    --split "$SPLIT" \
    --role "$ROLE" \
    # --sys \
    # --summarize \
    # --token "$TOKEN" \
    # --api \
