# import openai
import json
import numpy as np
import random
import requests
import time
from transformers import AutoTokenizer, GemmaTokenizer, AutoModelForCausalLM
# Use a pipeline as a high-level helper
from tqdm import tqdm
from transformers import pipeline
import transformers
import torch
import argparse


def generate_answer(answer_context, API=False):
    if not API:
        prompt = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
        # print(prompt)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(outputs)
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

def construct_assistant_message(completion, API=False):
    # content = completion["choices"][0]["message"]["content"]
    if API:
        content = completion[0]["generated_text"]
    else:
        content = completion
    content = content.split('model\n')[-1]
    # print(content)
    # return {"role": "assistant", "content": content}
    return {"role": "model", "content": content}

def summarize_message(agent_contexts, question, idx):
    if len(agent_contexts) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}
    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agent_contexts:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Write a summary of the different opinions from each of the individual agent."
    completion = generate_answer(prefix_string, API).split('model\n')[-1]
    prefix_string = f"Here is a summary of solutions from other agents: \n\n{completion}"
    prefix_string = prefix_string + """\n\n Use this summarization carefully as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)

    return {"role": "user", "content": prefix_string}

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def main(args):
# if __name__ == "__main__":
    
    agents = args.agents
    rounds = args.rounds
    random.seed(0)

    generated_description = {}

    questions = read_jsonl("grade-school-math/grade_school_math/data/test.jsonl")
    random.shuffle(questions)

    # for data in tqdm(questions[:100], desc='Answering questions'):
    for data in tqdm(questions, desc='Answering questions'):
        question = data['question']
        answer = data['answer']

        agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)}] for agent in range(agents)]

        for round in tqdm(range(rounds), desc='Rounds', leave=False):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    if args.summarize:
                        message = summarize_message(agent_contexts_other, question, 2*round - 1)
                    else:
                        message = construct_message(agent_contexts_other, question, 2*round - 1)
                    agent_context.append(message)

                # completion = openai.ChatCompletion.create(
                #           model="gpt-3.5-turbo-0301",
                #           messages=agent_context,
                #           n=1)
                completion = generate_answer(agent_context, args.api)

                assistant_message = construct_assistant_message(completion, args.api)
                agent_context.append(assistant_message)

        generated_description[question] = (agent_contexts, answer)

    json.dump(generated_description, open("gsm_{}_{}.json".format(agents, rounds), "w"))# DEBUG tensor can't be pickled

    # import pdb
    # pdb.set_trace()
    # print(answer)
    # print(agent_context)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate GSM data')
    parser.add_argument('--agents', type=int, default=3, help='Number of agents')
    parser.add_argument('--rounds', type=int, default=2, help='Number of rounds')
    parser.add_argument('--api', action='store_true', help='Use API', default=False)
    parser.add_argument('--model_id', type=str, default="google/gemma-2b-it", help='Model ID')
    parser.add_argument('--token', type=str, default="", help='API token')
    parser.add_argument('--summarize', action='store_true', help='Summarize', default=False)
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
        dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            torch_dtype=dtype,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    main(args)
