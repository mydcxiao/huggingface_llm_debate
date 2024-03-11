# import openai
import json
import numpy as np
import time
import pickle
from tqdm import tqdm
import argparse
import torch
import transformers
from transformers import AutoTokenizer, GemmaTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import pipeline


def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def generate_answer(answer_context, API=False):
    if not API:
        prompt = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
        # print(prompt)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        # outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=100, num_beams=4, do_sample=True)
        outputs = model.generate(input_ids=inputs.to(model.device), do_sample=True)
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
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

    # Use introspection in the case in which there are no other agents.
    if len(agents) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

    prefix_string = "These are the recent/updated opinions from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response.".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion, role, split, API=False):
    # content = completion["choices"][0]["message"]["content"]
    # return {"role": "assistant", "content": content}
    if API:
        content = completion[0]["generated_text"]
    else:
        content = completion
    content = content.split(split)[-1].strip()
    return {"role": f"{role}", "content": content}


def summarize_message(agent_contexts, question, idx, role, split, sys):
    
    if len(agent_contexts) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}
    
    prefix_string = "These are the recent/updated opinions from other agents: "

    for agent in agent_contexts:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Write a summary of the different opinions from each of the individual agent."
    context = [{"role": "user", "content": prefix_string}]
    if sys:
        context = [{"role": "system", "content": "You are a helpful assistant to summarize opinions from different agents in short and brief manner."}] + context
    completion = construct_assistant_message(generate_answer(context, API), role, split, API)['content']
    prefix_string = f"Here is a summary of recent/updated opinions from other agents: \n\n{completion}"
    prefix_string = prefix_string + "\n\n Use this summary carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response.".format(question)

    return {"role": "user", "content": prefix_string}


def parse_answer(sentence):
    parts = sentence.split(" ")

    for part in parts[::-1]:
        try:
            answer = float(part)
            return answer
        except:
            continue


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


def main(args):
    # answer = parse_answer("My answer is the same as the other agents and AI language model: the result of 12+28*19+6 is 550.")

    agents = args.agents
    rounds = args.rounds
    np.random.seed(0)

    evaluation_round = 100
    scores = []

    generated_description = {}

    for round in tqdm(range(evaluation_round), desc='Computing'):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)

        answer = a + b * c + d - e * f
        agent_contexts = [[{"role": "user", "content": """What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.""".format(a, b, c, d, e, f)}] for agent in range(agents)]

        if args.sys:
            sys_contexts = {"role": "system", "content": "You are a helpful assistant to solve arithmetic problems. Please think step by step and answer the questions in logical and brief manner."}
            agent_contexts = [[sys_contexts] + agent_context for agent_context in agent_contexts]
            
        # content = agent_contexts[0][0]['content']
        question_prompt = "We seek to find the result of {}+{}*{}+{}-{}*{}?".format(a, b, c, d, e, f)

        for round in tqdm(range(rounds), desc='Rounds', leave=False):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    idx = 2*round if args.sys else 2*round - 1
                    if args.summarize:
                        message = summarize_message(agent_contexts_other, question_prompt, idx, args.role, args.split, args.sys)
                    else:
                        message = construct_message(agent_contexts_other, question_prompt, idx)
                    agent_context.append(message)

                    # print("message: ", message)

                completion = generate_answer(agent_context, args.api)

                assistant_message = construct_assistant_message(completion, args.role, args.split, args.api)
                agent_context.append(assistant_message)
                # print(completion)

        text_answers = []
    
        lines = []
        for agent_context in agent_contexts:
            # text_answer = string =  agent_context[-1]['content']
            text_answer = agent_context[-1]['content']
            text_answer = text_answer.replace(",", ".")
            text_answer = parse_answer(text_answer)

            if text_answer is None:
                continue

            text_answers.append(text_answer)

        generated_description[(a, b, c, d, e, f)] = (agent_contexts, answer)

        try:
            text_answer = most_frequent(text_answers)
            if text_answer == answer:
                scores.append(1)
            else:
                scores.append(0)
        except:
            continue
            
        # print("performance:", np.mean(scores), np.std(scores) / (len(scores) ** 0.5))
        lines.append("performance: {} {}".format(np.mean(scores), np.std(scores) / (len(scores) ** 0.5)))
        
    with open("math_{}_{}.txt".format(agents, rounds), "a") as f:
        f.write("\n".join(lines))
    f.close()

    json.dump(generated_description, open("math_{}_{}.json".format(agents, rounds), "w"))
    print("performance:", np.mean(scores), np.std(scores) / (len(scores) ** 0.5))
    
    # pickle.dump(generated_description, open("math_agents{}_rounds{}.p".format(agents, rounds), "wb"))
    # import pdb
    # pdb.set_trace()
    # print(answer)
    # print(agent_context)
    
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
        torch_dtype = config.torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map='auto',
            torch_dtype=torch_dtype,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    main(args)
