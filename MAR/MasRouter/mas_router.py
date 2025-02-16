from typing import List, Dict
import os
import json
import torch
import torch.nn.functional as F
from torch.special import gammaln
import asyncio

from MAR.LLM.llm_embedding import SentenceEncoder
from MAR.Graph.graph import Graph
from MAR.Utils.utils import get_kwargs
from MAR.Utils.globals import Cost
from loguru import logger

class MasRouter(torch.nn.Module):
    """
    Input: Text descriptions of queries, tasks, LLMs, reasoning methods, roles, and corresponding tools
    Output: Task classification, number and types of LLMs required for each query, recommended reasoning methods and roles
    Description: LLMs include chatgpt, gemini, llama, etc., reasoning methods include single-agent CoT reasoning, multi-agent debate reasoning, multi-agent collaboration reasoning based on certain topological structures, roles include various identities, and various tools can be used, such as python compilers, wiki searches, etc.
    Requirements: Build a trainable model to construct the optimal multi-agent reasoning system
    """
    def __init__(self, in_dim:int = 384, hidden_dim:int = 16, max_agent:int = 6, device=None):
        """
        query: N*d tensor, N is the number of queries, d is the dimension of each query
        task: N_t*d tensor, N_t is the number of tasks, d is the dimension of each task
        llm: N_l*d tensor, N_l is the number of llm, d is the dimension of each llm
        """
        super().__init__()
        self.text_encoder = SentenceEncoder()
        self.task_classifier = TaskClassifier(input_dim = in_dim, hidden_dim = hidden_dim)
        self.llm_allocation = DynamicLLMAllocation(max_agent = max_agent)
        self.reasoning_selector = ReasoningSelector()
        self.role_selector = RoleSelector()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task_role_database, self.task_role_emb = self.encoder_roles()

    def forward(self, queries:List[str], tasks:List[Dict[str, str]], 
                llms: List[Dict[str, str]], reasonings:List[Dict[str, str]], given_task:List[int]=None, 
                prompt_file:str='MAR/Roles/FinalNode/gsm8k.json'):
        """
        queries:List[Dict[str, str]]: List of queries
        tasks:List[Dict[str, str]]: List of tasks
        llms:List[Dict[str, str]]: List of llms
        reasonings:List[Dict[str, str]]: List of reasonings
        """

        tasks_list = self._preprocess_data(tasks)
        llms_list = self._preprocess_data(llms)
        reasonings_list = self._preprocess_data(reasonings)

        queries_embedding = self.text_encoder(queries).to(self.device) 
        tasks_embedding = self.text_encoder(tasks_list).to(self.device) 
        llms_embedding = self.text_encoder(llms_list).to(self.device) 
        reasonings_embedding = self.text_encoder(reasonings_list).to(self.device)

        tasks_probs = self.task_classifier(queries_embedding, tasks_embedding)
        tasks_idx = torch.argmax(tasks_probs, dim=1) 
        selected_tasks_emb = tasks_embedding[tasks_idx] 
        selected_tasks:List[Dict[str,str]] = [tasks[idx] for idx in tasks_idx] if given_task is None else [tasks[idx] for idx in given_task]

        llms_num, log_probs = self.llm_allocation(queries_embedding, selected_tasks_emb, llms_embedding)
        selected_llms_emb = llms_num @ llms_embedding / llms_num.sum(dim=1).unsqueeze(1)
        selected_llms:List[List[Dict[str,str]]] = [[llms[idx] for idx, num in enumerate(row) for _ in range(int(num))] for row in llms_num] 

        selected_reason_idx, reason_log_probs = self.reasoning_selector(queries_embedding, selected_tasks_emb, selected_llms_emb, reasonings_embedding) 
        selected_reasons_emb = reasonings_embedding[selected_reason_idx] 
        log_probs = log_probs + reason_log_probs
        selected_reasons:List[Dict[str,str]] = [reasonings[idx] for idx in selected_reason_idx] 

        selected_roles, role_log_probs = self.role_selector(queries_embedding, selected_tasks_emb, selected_tasks,
                                                             llms_embedding, llms_num, 
                                                             selected_reasons_emb, self.task_role_database, self.task_role_emb) 
        log_probs = log_probs + role_log_probs

        final_result = []
        costs = []
        for query, task, llms, reason, roles in zip(queries, selected_tasks, selected_llms, selected_reasons, selected_roles):
            previous_cost = Cost.instance().value
            kwargs = get_kwargs(reason['Name'], len(llms))
            llm_names = [llm['Name'] for llm in llms]
            role_names = [role['Name'] for role in roles]
            logger.info(f'Query: {query}')
            logger.info(f'Task: {task["Name"]}')
            logger.info(f'LLMs: {llm_names}')
            logger.info(f'Reasoning: {reason["Name"]}')
            logger.info(f'Roles: {role_names}')
            logger.info('-----------------------------------')
            g = Graph(domain = task['Name'], llm_names = llm_names, agent_names = role_names, 
                      decision_method = "FinalRefer", prompt_file = prompt_file, reasoning_name=reason["Name"], **kwargs)
            final_result.append(g.run(inputs={"query":query}, num_rounds=kwargs["num_rounds"])[0][0])
            costs.append(Cost.instance().value - previous_cost)

        return final_result, costs, log_probs, tasks_probs
    
    async def aforward(self, queries: List[str], tasks: List[Dict[str, str]], 
                    llms: List[Dict[str, str]], reasonings: List[Dict[str, str]], given_task: List[int] = None, 
                    prompt_file: str = 'MAR/Roles/FinalNode/gsm8k.json'):
        tasks_list = self._preprocess_data(tasks)
        llms_list = self._preprocess_data(llms)
        reasonings_list = self._preprocess_data(reasonings)

        queries_embedding = self.text_encoder(queries).to(self.device)  
        tasks_embedding = self.text_encoder(tasks_list).to(self.device)  
        llms_embedding = self.text_encoder(llms_list).to(self.device)  
        reasonings_embedding = self.text_encoder(reasonings_list).to(self.device)  
        
        tasks_probs = self.task_classifier(queries_embedding, tasks_embedding)  
        tasks_idx = torch.argmax(tasks_probs, dim=1)  
        selected_tasks_emb = tasks_embedding[tasks_idx]  
        selected_tasks: List[Dict[str, str]] = [tasks[idx] for idx in tasks_idx] if given_task is None else [tasks[idx] for idx in given_task]
        
        llms_num, log_probs = self.llm_allocation(queries_embedding, selected_tasks_emb, llms_embedding) 
        selected_llms_emb = llms_num @ llms_embedding / llms_num.sum(dim=1).unsqueeze(1) 
        selected_llms: List[List[Dict[str, str]]] = [[llms[idx] for idx, num in enumerate(row) for _ in range(int(num))] for row in llms_num] 
        
        selected_reason_idx, reason_log_probs = self.reasoning_selector(queries_embedding, selected_tasks_emb, selected_llms_emb, reasonings_embedding)  
        selected_reasons_emb = reasonings_embedding[selected_reason_idx]  
        log_probs = log_probs + reason_log_probs
        selected_reasons: List[Dict[str, str]] = [reasonings[idx] for idx in selected_reason_idx] 
        
        selected_roles, role_log_probs = self.role_selector(queries_embedding, selected_tasks_emb, selected_tasks,
                                                        llms_embedding, llms_num, 
                                                        selected_reasons_emb, self.task_role_database, self.task_role_emb)  
        log_probs = log_probs + role_log_probs

        final_result = []
        costs = []
        
        async def process_query(query, task, llms, reason, roles):
            previous_cost = Cost.instance().value
            kwargs = get_kwargs(reason['Name'], len(llms))
            llm_names = [llm['Name'] for llm in llms]
            role_names = [role['Name'] for role in roles]
            logger.info(f'Query: {query}')
            logger.info(f'Task: {task["Name"]}')
            logger.info(f'LLMs: {llm_names}')
            logger.info(f'Reasoning: {reason["Name"]}')
            logger.info(f'Roles: {role_names}')
            logger.info('-----------------------------------')
            g = Graph(domain=task['Name'], llm_names=llm_names, agent_names=role_names, 
                    decision_method="FinalRefer", prompt_file=prompt_file, reasoning_name=reason["Name"], **kwargs)
            result = await g.arun(inputs={"query": query}, num_rounds=1)
            final_result.append(result[0][0])
            costs.append(Cost.instance().value - previous_cost)
        
        tasks = [process_query(query, task, llms, reason, roles) 
                for query, task, llms, reason, roles in zip(queries, selected_tasks, selected_llms, selected_reasons, selected_roles)]
        

        await asyncio.gather(*tasks)

        return final_result, costs, log_probs, tasks_probs

    def _preprocess_data(self, raw_data:List[Dict[str, str]]):
        get_name_description = lambda x: x['Name'] + ' : ' + x['Description']
        return [get_name_description(data) for data in raw_data]
    
    def encoder_roles(self):
        logger.info('Loading role embeddings...')
        task_role_database = {}
        task_role_emb = {}
        path = 'MAR/Roles'
        for task in os.listdir(path):
            task_path = os.path.join(path, task)
            if os.path.isdir(task_path):
                task_role_database[task] = []
                roles_list = []
                for role in os.listdir(task_path):
                    if role.endswith('.json'):
                        role_path = os.path.join(task_path, role)
                        role_profile = json.load(open(role_path, 'r', encoding='utf-8'))
                        task_role_database[task].append({"Name":role.split(".json")[0], "Desciption":role_profile})
                        roles_list.append(role_profile)
                task_role_emb[task] = self.text_encoder(roles_list).to(self.device)
        logger.info('Role embeddings loaded.')
        return task_role_database, task_role_emb
    
class TaskClassifier(torch.nn.Module):
    def __init__(self, input_dim:int=384, hidden_dim:int=32,device=None):
        super().__init__()
        self.query_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.task_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, queries, tasks):
        query_embedding = self.query_encoder(queries)
        task_embedding = self.task_encoder(tasks) 
        query_embedding = F.normalize(query_embedding, p=2, dim=1) 
        task_embedding = F.normalize(task_embedding, p=2, dim=1) 
        scores = torch.matmul(query_embedding, task_embedding.T) 
        return scores

class DynamicLLMAllocation(torch.nn.Module):
    def __init__(self, input_dim:int=384, hidden_dim:int=32, max_agent:int=1, device=None):
        super().__init__()
        self.max_agent = max_agent
        self.query_task_encoder = torch.nn.Linear(input_dim*2, hidden_dim)
        self.llm_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.difficulty_predictor = torch.nn.Linear(hidden_dim, 1)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, queries, tasks, llms):
        query_task_embedding = torch.cat([queries, tasks], dim=1) 
        query_task_embedding = self.query_task_encoder(query_task_embedding) 
        query_task_embedding = F.normalize(query_task_embedding, p=2, dim=1)
        llm_embedding = self.llm_encoder(llms) 
        llm_embedding = F.normalize(llm_embedding, p=2, dim=1)

        query_difficulty = self.difficulty_predictor(query_task_embedding) 
        query_difficulty = torch.sigmoid(query_difficulty) 
        llm_num_float = query_difficulty * self.max_agent 
        llm_num_int = torch.clamp(torch.round(llm_num_float), 1, self.max_agent).int() 

        scores = torch.matmul(query_task_embedding, llm_embedding.T) 
        scores = F.softmax(scores, dim=1)
        log_probs = torch.zeros_like(query_difficulty, device=llm_embedding.device)
        selected_llm = torch.zeros_like(scores, device=llm_embedding.device) 
        scores_cumsum = torch.cumsum(scores, dim=1)
        for i in range(1, self.max_agent+1):
            agent_num_mask = (llm_num_int >= i).squeeze(1).float() 
            random_num = torch.rand_like(llm_num_float, device=llm_embedding.device) 
            selected_index = (scores_cumsum > random_num).float().argmax(dim=1) 
            selected_llm[torch.arange(selected_llm.size(0)), selected_index] += agent_num_mask
        log_probs = log_probs + gammaln(llm_num_float + 1) - gammaln(selected_llm + 1).sum(dim=1).unsqueeze(1) + (selected_llm * torch.log(scores)).sum(dim=1).unsqueeze(1) # N_q*1
        return selected_llm, log_probs

class ReasoningSelector(torch.nn.Module):
    def __init__(self, input_dim:int=384, hidden_dim:int=32, device=None):
        super().__init__()
        self.query_task_llm_encoder = torch.nn.Linear(input_dim*3, hidden_dim)
        self.reasoning_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, queries, tasks, llms, reasonings):
        query_task_llm_embedding = torch.cat([queries, tasks, llms], dim=1) 
        query_task_llm_embedding = self.query_task_llm_encoder(query_task_llm_embedding) 
        query_task_llm_embedding = F.normalize(query_task_llm_embedding, p=2, dim=1)
        reasonings_embedding = self.reasoning_encoder(reasonings) 
        reasonings_embedding = F.normalize(reasonings_embedding, p=2, dim=1)
        scores = torch.matmul(query_task_llm_embedding, reasonings_embedding.T) 
        scores = torch.softmax(scores, dim=1) 
        scores_cumsum = torch.cumsum(scores, dim=1) 

        random_num = torch.rand([scores.size(0),1], device=reasonings_embedding.device) 
        selected_index = (scores_cumsum > random_num).float().argmax(dim=1) 
        log_probs = torch.log(scores[torch.arange(scores.size(0)), selected_index]).unsqueeze(1) 
        return selected_index, log_probs
    
class RoleSelector(torch.nn.Module):
    def __init__(self, input_dim:int=384, hidden_dim:int=32, device=None):
        super().__init__()
        self.qtlr_encoder = torch.nn.Linear(input_dim*4, hidden_dim)
        self.role_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, queries, tasks, selected_tasks, llms_embedding, llms_num, reasonings, role_database, role_emb):
        selected_roles = []
        log_probs = torch.zeros([queries.size(0),1], device=self.device)
        for q_i, q_llms_num in enumerate(llms_num):
            q_roles = []
            q_llms = [llms_embedding[l_i] for l_i, num in enumerate(q_llms_num) for _ in range(int(num))]
            for l_i, llm in enumerate(q_llms):
                llm_embedding = q_llms[l_i] # d
                qtlr_embedding = torch.cat([queries[q_i], tasks[q_i], llm_embedding, reasonings[q_i]], dim=0) 
                qtlr_embedding = self.qtlr_encoder(qtlr_embedding) 
                qtlr_embedding = F.normalize(qtlr_embedding, p=2, dim=0)
                role_embedding = role_emb[selected_tasks[q_i]['Name']] 
                role_embedding = self.role_encoder(role_embedding) 
                role_embedding = F.normalize(role_embedding, p=2, dim=1) 
                scores = torch.matmul(qtlr_embedding, role_embedding.T) 
                scores = torch.softmax(scores, dim=0) 
                scores_cumsum = torch.cumsum(scores, dim=0) 
                random_num = torch.rand([1], device=self.device) 
                selected_index = (scores_cumsum > random_num).float().argmax(dim=0) 
                log_probs[q_i,0] = log_probs[q_i,0] + torch.log(scores[selected_index])
                q_roles.append(role_database[selected_tasks[q_i]['Name']][selected_index])
            selected_roles.append(q_roles)
            
        return selected_roles, log_probs
    
