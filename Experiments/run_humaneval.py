import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import time
import argparse
import yaml
import json
import time
import re
import torch
from loguru import logger
import torch.nn.functional as F

from MAR.MasRouter.mas_router import MasRouter
from MAR.LLM.llm_profile import llm_profile
from MAR.Agent.reasoning_profile import reasoning_profile
from MAR.Prompts.tasks_profile import tasks_profile
from MAR.Tools.reader.readers import JSONLReader
from MAR.Tools.coding.python_executor import PyExecutor
from MAR.Utils.utils import fix_random_seed, split_list
from MAR.Utils.const import MAR_ROOT
from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
from MAR.Utils.log import configure_logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on humaneval")
    parser.add_argument("--dataset_json", type=str, default="Datasets/humaneval/humaneval-py.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.01,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4,help="batch size")
    parser.add_argument('--epochs', type=int, default=5, help="Default 5.")
    parser.add_argument('--num_rounds',type=int,default=1,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--domain', type=str, default="humaneval",help="Domain (the same as dataset name), default 'humaneval'")
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decison method of the agentprune')
    parser.add_argument('--prompt_file', type=str, default='MAR/Roles/FinalNode/humaneval.json')
    parser.add_argument('--start_epoch', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset = JSONLReader.parse_file("Datasets/humaneval/humaneval-py.jsonl")
    train_dataset, test_dataset = split_list(dataset, 0.2)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = f"humaneval_{current_time}.txt"
    fix_random_seed(1234)
    configure_logging(log_name=log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    router = MasRouter().to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=args.lr)
    tasks = tasks_profile
    llms = llm_profile
    reasonings = reasoning_profile

    logger.info("Start training...")
    for epoch in range(args.epochs):
        if epoch < args.start_epoch:
            router.load_state_dict(torch.load(f"humaneval_router_epoch{epoch}.pth", map_location=torch.device('cuda')))
            continue
        logger.info(f"Epoch {epoch}",80*'-')
        train_batches = int(len(train_dataset)/args.batch_size)
        total_solved, total_executed = (0, 0)
        for i_batch in range(train_batches):
            logger.info(f"Batch {i_batch}",80*'-')
            start_ts = time.time()
            current_batch = dataloader(train_dataset,args.batch_size,i_batch)
            queries = [item['prompt'] for item in current_batch]
            tests = [item['test'] for item in current_batch]
            task_labels = [2 for _ in current_batch]
            tasks_y = torch.tensor(task_labels).to(device)
            optimizer.zero_grad()
            results, costs, log_probs, tasks_probs = router.forward(queries, tasks, llms, reasonings, task_labels, prompt_file=args.prompt_file)

            task_loss = F.cross_entropy(tasks_probs, tasks_y)
            utilities = []
            answers_loss = []
            pattern = r'```python.*```'
            for query, result, test, log_prob, cost in zip(queries, results, tests, log_probs, costs):
                match = re.search(pattern, result, re.DOTALL|re.MULTILINE)
                if match:
                    answer = match.group(0).lstrip("```python\n").rstrip("\n```")
                    is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
                else:
                    answer = ""
                    is_solved = 0
                total_solved = total_solved + is_solved
                total_executed = total_executed + 1
                utility = is_solved - cost * 10
                utilities.append(utility)
                answer_loss = -log_prob * utility
                answers_loss.append(answer_loss)
                logger.debug(f"Raw Result: {result}")
                logger.debug(f"Answer: {answer}")
                logger.debug(f"Cost: {cost}")
            answer_loss = sum(answers_loss)/len(answers_loss)
            loss = task_loss + answer_loss
            loss.backward()
            optimizer.step()
        
            accuracy = total_solved / total_executed
            logger.info(f"Batch time {time.time() - start_ts:.3f}")
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"utilities:{utilities}")
            logger.info(f"task_loss:{task_loss.item()}" )
            logger.info(f"answer_loss:{answer_loss.item()}")
            logger.info(f"loss:{loss.item()}")
            logger.info(f"PromptTokens {PromptTokens.instance().value}")
            logger.info(f"CompletionTokens {CompletionTokens.instance().value}")
        logger.info(f"Epoch {epoch} Finishes",80*'-')
        torch.save(router.state_dict(), f"humaneval_router_epoch{epoch}.pth")
    logger.info("Finish training...")
    logger.info("Start testing...")
    test_batches = int(len(test_dataset)/args.batch_size)
    total_solved, total_executed = (0, 0)
    for i_batch in range(test_batches):
        logger.info(f"Batch {i_batch}",80*'-')
        start_ts = time.time()
        current_batch = dataloader(test_dataset,args.batch_size,i_batch)
        queries = [item['prompt'] for item in current_batch]
        tests = [item['test'] for item in current_batch]
        task_labels = [2 for _ in current_batch]
        tasks_y = torch.tensor(task_labels).to(device)
        results, costs, log_probs, tasks_probs = router.forward(queries, tasks, llms, reasonings, task_labels, prompt_file=args.prompt_file)

        utilities = []
        pattern = r'```python.*```'
        for query, result, test, log_prob, cost in zip(queries, results, tests, log_probs, costs):
            match = re.search(pattern, result, re.DOTALL|re.MULTILINE)
            if match:
                answer = match.group(0).lstrip("```python\n").rstrip("\n```")
                is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
            else:
                is_solved = 0
            total_solved = total_solved + is_solved
            total_executed = total_executed + 1
            logger.debug(f"Raw Result: {result}")
    
        accuracy = total_solved / total_executed
        logger.info(f"Batch time {time.time() - start_ts:.3f}")
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"PromptTokens {PromptTokens.instance().value}")
        logger.info(f"CompletionTokens {CompletionTokens.instance().value}")
    logger.info("Finish testing...")