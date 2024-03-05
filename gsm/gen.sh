export AGENTS=1
export ROUNDS=3
export MODEL_ID="meta-llama/Llama-2-7b-chat-hf"
export TOKEN=""
export SPLIT="[/INST]" #"model\n"
export ROLE="assistant" #"model"
export CUDA_VISIBLE_DEVICES=0

python gen_gsm.py \
    --agents $AGENTS \
    --rounds $ROUNDS \
    --model_id "$MODEL_ID" \
    --split "$SPLIT" \
    --role "$ROLE" \
    --sys \
    # --summarize \
    # --token "$TOKEN" \
    # --api \
