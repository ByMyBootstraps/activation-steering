import os
from datetime import datetime

def fancy_round(number, n=2):
    if number == 0:
        return 0
    
    if abs(number) > 1:
        return round(number, n)
    
    # Find the number of zeros after the decimal point until the first non-zero
    number_str = '{:f}'.format(number)
    decimal = number_str.split('.')[1]
    num_zeros = len(decimal) - len(decimal.lstrip('0'))
    precision = num_zeros + n

    # Choose the maximum between the required n decimal places and the calculated precision
    precision = max(n, precision)
    return round(number, precision)

def pad(d, n=2):
    s = str(d)
    if len(s) < n:
        return "0" * (n - len(s)) + s
    return d

DATA_ROOT = "./"
def recordSpending( model: str, prompt_tokens: int, completion_tokens: int, logs_dir="" ):
    if model.startswith("gpt-3.5-turbo-16k"):
        ppprompt = 0.003 / 1000
        ppcompletion = 0.004 / 1000
    elif model.startswith( "gpt-3.5-turbo" ):
        ppprompt = 0.0015 / 1000
        ppcompletion = 0.002  / 1000
    elif model.startswith( "gpt-4-1106-preview" ):
        ppprompt = 0.01 / 1000
        ppcompletion = 0.03 / 1000
    elif model.startswith( "gpt-4-32k" ):
        ppprompt = 0.06 / 1000
        ppcompletion = 0.12 / 1000
    elif model.startswith( "gpt-4" ):
        ppprompt = 0.03 / 1000
        ppcompletion = 0.06  / 1000
    else:
        raise Exception( "Unknown model" )
    
    spending_prompt, spending_completion = ppprompt * prompt_tokens, ppcompletion * completion_tokens

    if logs_dir == "":
        logs_dir = os.path.join(DATA_ROOT, "logs/")

        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
    records_file = os.path.join(logs_dir, "spending_records.txt")

    with open( records_file, "a" ) as f:
        f.write( f"{datetime.now()}\t{model}\t{prompt_tokens}\t{spending_prompt}\t{completion_tokens}\t{spending_completion}\n" )

    print( f"Prompt tokens: {prompt_tokens}, completion tokens: {completion_tokens}" )
    print( f"Spent {fancy_round(spending_prompt)} on prompt, {fancy_round(spending_completion)} on completion" )
    print( f"Total API cost: {fancy_round(spending_prompt + spending_completion)}" )
    
    # print( f"Total API spending today: ..." )
