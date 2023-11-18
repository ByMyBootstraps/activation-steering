from datasets import load_dataset, Dataset
import torch
import requests
import json
from random import sample

from utils.model import buildTokenizer

def generate_and_tokenize_prompt(transcript, tokenizer, add_generation_prompt=False):
    text = tokenizer.apply_chat_template(transcript, tokenize=False, add_generation_prompt=add_generation_prompt)

    return tokenizer.encode(text, return_tensors="pt")[0]

def tokenize_score(score, tokenizer):
    if score == "A":
        return torch.Tensor([28741]).to(torch.int32)[0]
    elif score == "B":
        return torch.Tensor([28760]).to(torch.int32)[0]
    else:
        raise Exception("Invalid score")
    # return tokenizer.encode(str(score), return_tensors="pt")[0][1]

def buildTinyContrastiveDataset(data):
    tokenizer = buildTokenizer()
    def tokenizePrompt( prompt ):
        sts = [
            {
                "role": "system",
                "content": "You are helpful, honest, and concise assistant"
            },
            {
                "role": "user",
                "content": prompt["question"]
            },
            {
                "role": "assistant",
                "content": prompt["answer"]
            }
        ]
        cts = [
            {
                "role": "system",
                "content": "You are helpful, honest, and concise assistant"
            },
            {
                "role": "user",
                "content": prompt["question"]
            },
            {
                "role": "assistant",
                "content": prompt["contrastive_answer"]
            }
        ]

        label = "A" # collating labels wasn't working so I infer them later

        stokens = generate_and_tokenize_prompt(sts, tokenizer)
        ctokens = generate_and_tokenize_prompt(cts, tokenizer)
        score_token = tokenize_score(label, tokenizer)

        return [stokens, ctokens, score_token]
    
    new_data = [tokenizePrompt(prompt) for prompt in data]

    sdataset = Dataset.from_dict({
        "input_ids": [item[0] for item in new_data],
        "labels": [item[2] for item in new_data]
    })
    cdataset = Dataset.from_dict({
        "input_ids": [item[1] for item in new_data],
        "labels": [item[2] for item in new_data]
    })
    # Set format for PyTorch
    sdataset.set_format(type='torch', columns=['input_ids', 'labels'])
    cdataset.set_format(type='torch', columns=['input_ids', 'labels'])
    return sdataset, cdataset

# from beginning is a silly hack to make sure my train and eval subsets are disjoint
def get_sycophancy_data(n_samples, from_beginning=True):
    DATASETS = [
        "sycophancy_on_nlp_survey.jsonl",
        "sycophancy_on_philpapers2020.jsonl",
        "sycophancy_on_political_typology_quiz.jsonl",
    ]
    all_data = []
    for item in DATASETS:
        url = f"https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/sycophancy/{item}"
        r = requests.get(url).text
        data = [json.loads(l) for l in r.split("\n") if l != ""]
        for d in data:
            try:
                assert d["question"].split("\n\n")[-1] == "Answer:"
            except:
                print( d["question"] )
            all_data.append(
                {
                    "sycophantic_completion": d["answer_matching_behavior"].strip()[1:-1],
                    "non_sycophantic_completion": d["answer_not_matching_behavior"].strip()[1:-1],
                    "question": "\n\n".join( d["question"].split("\n\n")[:-1] ),
                }
            )
    print( len(all_data) )
    return all_data[:n_samples] if from_beginning else all_data[-n_samples:]

def get_sycophancy_dataset(n_samples, from_beginning=False):
    tokenizer = buildTokenizer()
    data = get_sycophancy_data(n_samples, from_beginning=from_beginning)
    def buildPrompt( datum ):
        ts = [
            {
                "role": "system",
                "content": "You are helpful, honest, and concise assistant"
            },
            {
                "role": "user",
                "content": datum["question"] + """

Output A or B with no additional formatting or commentary."""
            }
        ]
        return {
            "transcript": ts,
            "sycophantic_completion": datum["sycophantic_completion"],
            "non_sycophantic_completion": datum["non_sycophantic_completion"],
            "sycophantic_transcript": ts + [{"role": "assistant", "content": datum["sycophantic_completion"]}],
            "non_sycophantic_transcript": ts + [{"role": "assistant", "content": datum["non_sycophantic_completion"]}],
            "question": datum["question"]
        }
    data = [buildPrompt(datum) for datum in data]
    dataset = []
    for datum in data:
        tokens = generate_and_tokenize_prompt(datum["transcript"], tokenizer, add_generation_prompt=True)
        s_tokens = generate_and_tokenize_prompt(datum["sycophantic_transcript"], tokenizer)
        n_tokens = generate_and_tokenize_prompt(datum["non_sycophantic_transcript"], tokenizer)
        score_token = tokenize_score(datum["sycophantic_completion"], tokenizer)
        question = datum["question"]
        prompt = datum["transcript"]
        solution = datum["sycophantic_completion"]
        dataset.append( [tokens, score_token, s_tokens, n_tokens, question, prompt, solution] )

    dataset = Dataset.from_dict({
        "input_ids": [item[0] for item in dataset],
        "labels": [item[1] for item in dataset],
        "s_tokens": [item[2] for item in dataset],
        "n_tokens": [item[3] for item in dataset],
        "questions": [item[4] for item in dataset],
        "prompt": [item[5] for item in dataset],
        "solution": [item[6] for item in dataset],
    })

    # Set format for PyTorch
    dataset.set_format(type='torch', columns=['input_ids', 'labels', "s_tokens", "n_tokens", "questions", "prompt", "solution"])

    return dataset

def contrastive_sycophancy_dataset(n_samples, from_beginning=True):
    data = get_sycophancy_data(n_samples)
    data = [{
        "question": datum["question"],
        "answer": datum["sycophantic_completion"],
        "contrastive_answer": datum["non_sycophantic_completion"]
    } for datum in data]
    return buildTinyContrastiveDataset(data)

def getTruthfulQARaw():
    questions = []
    for i in range(10):
        url = f"https://datasets-server.huggingface.co/rows?dataset=truthful_qa&config=generation&split=validation&offset={i * 100}&limit=100"
        r = requests.get(url).text
        t_qa_data = [row["row"] for row in json.loads(r)["rows"]]
        for d in t_qa_data:
            prompt = [
                {
                    "role": "system",
                    "content": "You are helpful, honest, and concise assistant"
                },
                {
                    "role": "user",
                    "content": d["question"]
                }
            ]
            solution = d["best_answer"]
            questions.append( {
                "prompt": prompt,
                "solution": solution,
                "question": d["question"]
            } )
    return questions