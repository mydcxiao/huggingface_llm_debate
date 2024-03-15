# import openai
import json
import numpy as np
import random
import requests
import time
import argparse
import os
from tqdm import tqdm
import torch
import transformers
from transformers import AutoTokenizer, GemmaTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
from transformers import pipeline


def generate_answer(answer_context, API=False):
    if not API:
        prompt = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(input_ids=inputs.to(model.device), generation_config=gen_config)
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return outputs
    else:
        try:
            prompt = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
            payload = {"inputs": prompt}
            # payload = {"inputs": "".join(answer_context),
            #            "parameters": {
            #                 "max_new_tokens": 256
            #             }
            #            }
            completion = requests.post(API_URL, headers=headers, json=payload)
            completion = completion.json()
            
        except Exception as e:
            print(f"retrying due to an error......{e}")
            time.sleep(5)
            return generate_answer(answer_context)

        return completion

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}

def construct_assistant_message(completion, role, split, API=False):
    # content = completion["choices"][0]["message"]["content"]
    if API:
        content = completion[0]["generated_text"]
    else:
        content = completion
    content = content.split(split)[-1].strip()
    return {"role": f"{role}", "content": content}

def summarize_message(agent_contexts, question, idx, role, split, sys):
    if len(agent_contexts) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}
    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agent_contexts:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Write a summary of the different opinions from each of the individual agent in brief and logical manner."
    context = [{"role": "user", "content": prefix_string}]
    if sys:
        context = [{"role": "system", "content": "You are a helpful assistant to summarize opinions from different agents in brief and logical manner."}] + context
    completion = construct_assistant_message(generate_answer(context, API), role, split, API)['content']
    prefix_string = f"Here is a summary of solutions from other agents: \n\n{completion}"
    prefix_string = prefix_string + """\n\n Use this summary carefully as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)

    return {"role": "user", "content": prefix_string}

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def main(args):
    
    agents = args.agents
    rounds = args.rounds
    random.seed(0)

    generated_description = {}

    questions = read_jsonl("grade-school-math/grade_school_math/data/test.jsonl")
    random.shuffle(questions)

    for data in tqdm(questions[:1], desc='Answering questions'):
        question = data['question']
        answer = data['answer']

        agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)}] for agent in range(agents)]
        
        # for models with <sys> roles
        if args.sys:
            sys_contexts = {"role": "system", "content": "You are a helpful assistant to solve graduate school math problems. Please think step by step and answer the questions in logical and brief manner."}
            agent_contexts = [[sys_contexts] + agent_context for agent_context in agent_contexts]
        
        for round in tqdm(range(rounds), desc='Rounds', leave=False):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    idx = 2*round if args.sys else 2*round - 1
                    if args.summarize:
                        message = summarize_message(agent_contexts_other, question, idx, args.role, args.split, args.sys)
                    else:
                        message = construct_message(agent_contexts_other, question, idx)
                    agent_context.append(message)

                # completion = openai.ChatCompletion.create(
                #           model="gpt-3.5-turbo-0301",
                #           messages=agent_context,
                #           n=1)
                completion = generate_answer(agent_context, args.api)

                assistant_message = construct_assistant_message(completion, args.role, args.split, args.api)
                agent_context.append(assistant_message)

        generated_description[question] = (agent_contexts, answer)

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(generated_description, open(os.path.join(args.output_dir, "gsm_{}_{}.json".format(agents, rounds)), "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate GSM data')
    parser.add_argument('--agents', type=int, default=3, help='Number of agents')
    parser.add_argument('--rounds', type=int, default=2, help='Number of rounds')
    parser.add_argument('--api', action='store_true', help='Use API', default=False)
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='Model ID')
    parser.add_argument('--token', type=str, default="", help='API token')
    parser.add_argument('--summarize', action='store_true', help='Summarize', default=False)
    parser.add_argument('--sys', action='store_true', help='Use sys role', default=False)
    parser.add_argument('--split', type=str, default="[/INST]", help='Response split')
    parser.add_argument('--role', type=str, default="assistant", help='Role')
    parser.add_argument('--output_dir', type=str, default="output/", help='Output directory')
    
    args = parser.parse_args()
    API = args.api
    model_id = args.model_id
    # pipe = pipeline("text-generation", model=model_id)
    if API:
        YOUR_TOKEN = args.token
        API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {YOUR_TOKEN}",
                "Content-Type": "application/json"}
    else:
        config = AutoConfig.from_pretrained(model_id)
        gen_config = GenerationConfig.from_pretrained(model_id)
        if gen_config.max_length == 20:
            gen_config.max_length = 4096
        gen_config.do_sample = True
        gen_config.pad_token_id = gen_config.pad_token_id if hasattr(gen_config, "pad_token_id") and gen_config.pad_token_id else \
            config.pad_token_id if hasattr(config, "pad_token_id") and config.pad_token_id else 0
        torch_dtype = config.torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch_dtype,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    main(args)
