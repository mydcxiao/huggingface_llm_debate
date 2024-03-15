# Open-source LLM Multiagent Debate

This repository is the implementation of [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) based on Huggingface open-source LLM models.

Some code and prompts are borrowed from [original paper released code](https://composable-models.github.io/llm_debate/) and [LLM Agora](https://github.com/gauss5930/LLM-Agora/) 
You may see some additional debate logs from original release [here](https://www.dropbox.com/sh/6kq5ixfnf4zqk09/AABezsYsBhgg1IQAZ12yQ43_a?dl=0).

## TODO:
✅ Math huggingface open-source LLM debate.
✅ GSM8K huggingface open-source LLM debate.
- [ ] MMLU huggingface open-source LLM debate.
- [ ] Biography huggingface open-source LLM debate.

## Running experiments

The code for running arithmetic, GSM, biographies, and MMLU tasks may be found in the following subfolders

* ./math/ contains code for running math
* ./gsm/ contains code for running gsm
* ./biography/ contains code for running biographies
* ./mmlu/ contains code for running mmlu results.

**Math:**

To generate and evaluated answer for Math problems through multiagent debate, cd into the math directory:
- run on local model:
```shell 
python gen_math.py \
    --agents 3 \
    --rounds 2 \
    --model_id "meta-llama/Llama-2-7b-chat-hf" \
    --split "[/INST]" \
    --role "assistant" \
    --output_dir "output" \
    --summarize \
    --sys
```
`--summarize` will enable summarization throughout debate mentioned in the paper.

`--sys` will append system message before the message for models capable of system message.

`--model_id` is the model dir in huggingface hub.

`--split` is the special token used to split question and response in multi-round chat model.

`--role` is the role of message for multi-round chat model.

- run using huggingface API:
```shell
python gen_math.py \
    --agents 3 \
    --rounds 2 \
    --model_id "meta-llama/Llama-2-7b-chat-hf" \
    --split "[/INST]" \
    --role "assistant" \
    --output_dir "output" \
    --summarize \
    --sys \
    --api \
    --token "dhqieiq"
```
`--api` will let the program to fetch reponses from huggingface API.

`--token` input your huggingface token here.
	
**Grade School Math:**

To generate answers for Grade School Math problems through multiagent debate, cd into the gsm directory:
- run on local model:
```shell 
python gen_gsm.py \
    --agents 3 \
    --rounds 2 \
    --model_id "meta-llama/Llama-2-7b-chat-hf" \
    --split "[/INST]" \
    --role "assistant" \
    --output_dir "output" \
    --summarize \
    --sys
```
`--summarize` will enable summarization throughout debate mentioned in the paper.

`--sys` will append system message before the message for models capable of system message.

`--model_id` is the model dir in huggingface hub.

`--split` is the special token used to split question and response in multi-round chat model.

`--role` is the role of message for multi-round chat model.

- run using huggingface API:
```shell
python gen_gsm.py \
    --agents 3 \
    --rounds 2 \
    --model_id "meta-llama/Llama-2-7b-chat-hf" \
    --split "[/INST]" \
    --role "assistant" \
    --output_dir "output" \
    --summarize \
    --sys \
    --api \
    --token "dhqieiq"
```
`--api` will let the program to fetch reponses from huggingface API.

`--token` input your huggingface token here.

To evaluate the generated results of Grade School Math problems:
```shell
python eval_gsm.py --path 'gsm_3_2.json'
```
`--path` is the saved json response file path from gen_gsm.py
 
You can download the GSM dataset [here](https://github.com/openai/grade-school-math)


**Biography:**

To generate answers for Biography problems through multiagent debate, cd into the biography directory and run:
	`python gen_conversation.py`

To evaluate the generated results for Biography problems:
	`python eval_conversation.py`
	
**MMLU:**

To generate answers for MMLU through multiagent debate, cd into the MMLU directory and run:
	`python gen_mmlu.py`

To evaluate the generated results of MMLU:
	`python eval_mmlu.py`
	
You can download the MMLU dataset [here](https://github.com/hendrycks/test)
