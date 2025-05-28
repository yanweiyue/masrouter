from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
import tiktoken
# GPT-4:  https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
# GPT3.5: https://platform.openai.com/docs/models/gpt-3-5
# DALL-E: https://openai.com/pricing

def cal_token(model:str, text:str):
    encoder = tiktoken.encoding_for_model('gpt-4o')
    num_tokens = len(encoder.encode(text))
    return num_tokens

def cost_count(prompt, response, model_name):
    prompt_len: int
    completion_len: int
    price: float

    prompt_len = cal_token(model_name, prompt)
    completion_len = cal_token(model_name, response)
    if model_name not in MODEL_PRICE.keys():
        return 0, 0, 0
    prompt_price = MODEL_PRICE[model_name]["input"]
    completion_price = MODEL_PRICE[model_name]["output"]
    price = prompt_len * prompt_price / 1000000 + completion_len * completion_price / 1000000

    Cost.instance().value += price
    PromptTokens.instance().value += prompt_len
    CompletionTokens.instance().value += completion_len

    # print(f"Prompt Tokens: {prompt_len}, Completion Tokens: {completion_len}")
    return price, prompt_len, completion_len

MODEL_PRICE = {
    "gpt-3.5-turbo-0125":{
        "input": 0.5,
        "output": 1.5
    },
    "gpt-3.5-turbo-1106":{
        "input": 1.0,
        "output": 2.0
    },
    "gpt-4-1106-preview":{
        "input": 10.0,
        "output": 30.0
    },
    "gpt-4o":{
        "input": 2.5,
        "output": 10.0
    },
    "gpt-4o-mini":{
        "input": 0.15,
        "output": 0.6
    },
    "claude-3-5-haiku-20241022":{
        "input": 0.8,
        "output": 4.0
    },
    "claude-3-5-sonnet-20241022":{
        "input": 3.0,
        "output": 15.0
    },
    "gemini-1.5-flash-latest":{
        "input": 0.15,
        "output": 0.60
    },
    "gemini-2.0-flash-thinking-exp":{
        "input": 4.0,
        "output": 16.0
    },
    "llama-3.3-70b-versatile":{
        "input": 0.2,
        "output": 0.2
    },
    "Meta-Llama-3.1-70B-Instruct":{
        "input": 0.2,
        "output": 0.2
    },
    "llama-3.1-70b-instruct":{
        "input": 0.2,
        "output": 0.2
    },
    'deepseek-chat':{
        'input': 0.27,
        'output': 1.1
    },
    'deepseek-ai/DeepSeek-V3':{
        'input': 0.27,
        'output': 1.1
    }
}
