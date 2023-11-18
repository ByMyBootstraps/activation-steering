import openai
from openai import APIConnectionError, APIError, Timeout

from utils.spend import recordSpending
def chat( messages, model, max_tokens, auto_retry=True):
    try:
        resp = openai.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens
        )
    except (APIConnectionError, APIError, Timeout) as e:
        if auto_retry:
            print( "APIConnectionError: retrying..." )
            return chat( messages, model, auto_retry=False )
        else:
            raise e
    except Exception as e:
        print( "IF THIS SHOULD HAVE AUTO-RETRIED, EDIT THIS CODE")
        raise e
    
    recordSpending( resp.model, resp.usage.prompt_tokens, resp.usage.completion_tokens )

    return resp    

def evaluate(question, answer1, answer2, metric):
    transcript1 = [
        {
            "role": "system",
            "content": "You are helpful, honest, and concise assistant"
        },
        {
            "role": "user",
            "content": f"""{metric}

The question is:
{question}

The first answer is:
{answer1}

The second answer is:
{answer2}"""
        }
    ]
    transcript2 = [
        {
            "role": "system",
            "content": "You are helpful, honest, and concise assistant"
        },
        {
            "role": "user",
            "content": f"""{metric}

The question is:
{question}

The first answer is:
{answer1}

The second answer is:
{answer2}

Your response must either be "1", "2", or "tie". Do not output any other tokens. Do not output any commentary."""
        }
    ]

    resp1 = chat(
        messages=transcript1,
        model="gpt-4",
        max_tokens=1
    )

    resp2 = chat(
        messages=transcript2,
        model="gpt-4",
        max_tokens=1
    )

    print( resp1 )

    r1 = resp1.choices[0].message.content
    r2 = resp2.choices[0].message.content

    c1 = 0
    c2 = 0
    
    if r1 == "1":
        c1 += 1
    elif r1 == "2":
        c2 += 1

    if r2 == "1":
        c1 += 1
    elif r2 == "2":
        c2 += 1

    if c1 > c2:
        return 1
    elif c2 > c1:
        return 2
    else:
        return 0

    return resp1, resp2