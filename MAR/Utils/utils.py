import re
import torch
import shortuuid
from collections import Counter
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from typing import List, Union, Literal, Optional

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "The answer is"

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

def nuclear_norm(matrix):
    _, S, _ = torch.svd(matrix)
    return torch.sum(S)

def frobenius_norm(A, S):
    return torch.norm(A - S, p='fro')

used_ids = set()
def generate_unique_ids(n:int=1,pre:str="",length:int=4)->List[str]:
    ids = set()
    while len(ids) < n:
        random_id = shortuuid.ShortUUID().random(length=length)
        if pre:
            random_id = f"{pre}_{random_id}"
        if pre in used_ids:
            length += 1
            continue
        ids.add(random_id)
    return list(ids)

def extract_json(raw:str)->str:
    """
    Extract the json string from the raw string.
    If there is no json string, return an empty string.
    """
    json_pattern = r'\{.*\}' 
    match = re.search(json_pattern, raw, re.DOTALL)
    return match.group(0) if match else ""

def fix_random_seed(seed:int=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_mode(nums):
    count = Counter(nums)
    mode, _ = count.most_common(1)[0]
    return mode


def get_kwargs(mode:Union[Literal['DirectAnswer','CoT','IO','FullConnected','Random','Chain','Debate','Layered','Star'],str]
               ,N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks: Optional[List[List[int]]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:Optional[List[List[int]]] = None
    node_kwargs = None
    num_rounds = 1
    # agent_names = []

    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i+1,n):
                matrix[i][j] = 1
        return matrix
    
    if mode=='DirectAnswer' or mode=='CoT' or mode=='IO' or mode=='Reflection':
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[0 for _ in range(N)] for _ in range(N)]
    elif mode=='FullConnected':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
        num_rounds = 2
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]


    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs,
            "num_rounds":num_rounds,}

def split_list(input_list, ratio):
    if not (0 < ratio < 1):
        raise ValueError("Ratio must be between 0 and 1.")
    
    random.shuffle(input_list)
    split_index = int(len(input_list) * ratio)
    part1 = input_list[:split_index]
    part2 = input_list[split_index:]
    
    return part1, part2

def plot_embedding_heatmap(embedding: torch.Tensor, title: str, save_path: str):
    embedding_np = embedding.detach().cpu().numpy()

    plt.figure(figsize=(10, max(4, embedding_np.shape[0] * 0.4)))
    sns.heatmap(embedding_np, cmap="viridis", cbar=True)

    plt.title(title)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Index")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_row_similarity(embeddings: torch.Tensor, title: str, save_path: str):
    embeddings_np = embeddings.detach().cpu().numpy()
    row_similarities = np.corrcoef(embeddings_np)

    plt.figure(figsize=(10, max(4, row_similarities.shape[0] * 0.4)))
    sns.heatmap(row_similarities, cmap="viridis", cbar=True)

    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Index")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()