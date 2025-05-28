from typing import List, Dict, Optional
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import gammaln
import math

from MAR.LLM.llm_embedding import SentenceEncoder
from MAR.Graph.graph import Graph
from MAR.Utils.utils import get_kwargs, plot_embedding_heatmap, plot_row_similarity
from MAR.Utils.globals import Cost
from loguru import logger

class GFusion(nn.Module):
    def __init__(self, d_model:int=384):
        """
        Graph Fusion Module: GFM
        Input: x: [xx, d], y: [yy, d]
        Output: z: [xx, d]
        """
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, y):
        Q = self.query_proj(x)      # [xx, d]
        K = self.key_proj(y)        # [yy, d]
        V = self.value_proj(y)      # [yy, d]

        attn_scores = torch.matmul(Q, K.transpose(0, 1)) / (Q.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [xx, yy]
        context = torch.matmul(attn_weights, V)  # context: [xx, d]
        context = F.normalize(context, p=2, dim=1)
        z = self.out_proj(x + context)  # [xx, d]
        return z

std2 = 0.1
var2 = std2 * std2
log_var2 = math.log(var2)

class VAE(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=64, latent_dim=64):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)  # μ, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)*std2
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return self.fc4(h) # x_hat

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, z, mu, log_var

def vae_loss_function(x_hat, x, mu, log_var):
    MSE = F.mse_loss(x_hat, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 - log_var2 + log_var - (mu.pow(2) + log_var.exp())/var2)
    return MSE + KLD

class MasRouter(nn.Module):
    """
    Input: Text descriptions of queries, tasks, LLMs, collab methods, roles, and corresponding tools
    Output: Task classification, number and types of LLMs required for each query, recommended collab reasoning methods and roles
    Description: LLMs include chatgpt, gemini, llama, etc., collab reasoning methods include single-agent CoT reasoning, multi-agent debate reasoning, multi-agent collaboration reasoning based on certain topological structures, roles include various identities, and various tools can be used, such as python compilers, wiki searches, etc.
    Requirements: Build a trainable model to construct the optimal multi-agent system
    """
    def __init__(self, in_dim:int = 384, hidden_dim:int = 64, max_agent:int = 6, temp:float=0.5, device=None):
        """
        query: N*d tensor, N is the number of queries, d is the dimension of each query
        task: N_t*d tensor, N_t is the number of tasks, d is the dimension of each task
        llm: N_l*d tensor, N_l is the number of llm, d is the dimension of each llm
        """
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_encoder = SentenceEncoder(device=self.device)
        self.task_classifier = TaskClassifier(input_dim = in_dim, hidden_dim=hidden_dim, device=self.device,temp=temp)
        self.collab_determiner = CollabDeterminer(input_dim = in_dim, context_input_dim = in_dim , hidden_dim = hidden_dim,device=self.device,temp=0.8)
        self.num_determiner = NumDeterminer(input_dim = in_dim, hidden_dim=hidden_dim,max_agent=max_agent, device=self.device)
        self.role_allocation = RoleAllocation(input_dim = in_dim, context_input_dim = 2* hidden_dim, hidden_dim=hidden_dim,device=self.device,temp=temp)
        self.llm_router = LLMRouter(device=self.device,max_agent=max_agent,temp=1.0)

    def forward(self, queries:List[str], tasks:List[Dict[str, str]], 
                llms: List[Dict[str, str]], collabs:List[Dict[str, str]], given_task: Optional[List[int]] = None, 
                prompt_file:str='MAR/Roles/FinalNode/gsm8k.json'):
        """
        queries:List[Dict[str, str]]: List of queries
        tasks:List[Dict[str, str]]: List of tasks
        llms:List[Dict[str, str]]: List of llms
        collabs:List[Dict[str, str]]: List of collabs
        """
        # Preprocess data
        tasks_list = self._preprocess_data(tasks)
        llms_list = self._preprocess_data(llms)
        collabs_list = self._preprocess_data(collabs)
        task_role_database, task_role_emb = self.encoder_roles() # task_role_database: Dict[str, List[Dict[str, str]]], task_role_emb: Dict[str, torch.Tensor]

        # Text embedding
        queries_embedding = self.text_encoder(queries) # N_q*d tensor, N_q is the number of queries, d is the dimension of each query
        tasks_embedding = self.text_encoder(tasks_list) # N_t*d tensor, N_t is the number of tasks, d is the dimension of each task
        llms_embedding = self.text_encoder(llms_list) # N_l*d tensor, N_l is the number of llms, d is the dimension of each llm
        collabs_embedding = self.text_encoder(collabs_list) # N_r*d tensor, N_r is the number of collabs, d is the dimension of each collab
        
        # Task classification
        selected_tasks_idx, tasks_probs, query_context = self.task_classifier(queries_embedding, tasks_embedding) # N_q, N_q*1，N_q*hidden_dim
        selected_tasks:List[Dict[str,str]] = [tasks[idx] for idx in selected_tasks_idx] if given_task is None else [tasks[idx] for idx in given_task]
        tasks_role_list:List[List[Dict[str,str]]] = [task_role_database[task['Name']] for task in selected_tasks]
        tasks_role_emb_list:List[torch.Tensor] = [task_role_emb[task['Name']] for task in selected_tasks]

        # Collaboration method selection
        selected_collabs_idx, collab_log_probs, collab_context, collab_vae_loss = self.collab_determiner(collabs_embedding, queries_embedding) # N_q, N_q*1，N_q*hidden_dim
        selected_collabs:List[Dict[str,str]] = [collabs[idx] for idx in selected_collabs_idx]
        
        # Number of agents determination
        agent_num_int, agent_num_float, num_vae_loss = self.num_determiner(queries_embedding) # N_q*1, N_q*1
        
        # Role selection
        selected_roles_idx, role_log_probs, role_context, role_vae_loss = self.role_allocation(tasks_role_emb_list, torch.concat([query_context, collab_context],dim=-1), agent_num_int) # N_q*agent_num, N_q*1，N_q*hidden_dim
        selected_roles:List[List[Dict[str,str]]] = [[tasks_roles[selected_role_id.item()] for selected_role_id in selected_roles_id_list] for tasks_roles, selected_roles_id_list in zip(tasks_role_list, selected_roles_idx)]
        
        # LLM allocation
        selected_llms_idx, llm_log_probs, llm_vae_loss = self.llm_router(llms_embedding, torch.concat([query_context, collab_context, role_context],dim=-1), agent_num_int, agent_num_float) # N_q*1，N_q*hidden_dim
        selected_llms:List[List[Dict[str,str]]] = [[llms[idx] for idx in selected_llms_id_list] for selected_llms_id_list in selected_llms_idx]
        log_probs = llm_log_probs + role_log_probs + collab_log_probs # N_q*1

        vae_loss = collab_vae_loss + num_vae_loss + role_vae_loss + llm_vae_loss

        final_result = []
        costs = []
        for query, task, llms, collab, roles in zip(queries, selected_tasks, selected_llms, selected_collabs, selected_roles):
            previous_cost = Cost.instance().value
            kwargs = get_kwargs(collab['Name'], len(llms))
            llm_names = [llm['Name'] for llm in llms]
            role_names = [role['Name'] for role in roles]
            logger.info(f'Query: {query}')
            logger.info(f'Task: {task["Name"]}')
            logger.info(f'LLMs: {llm_names}')
            logger.info(f'Reasoning: {collab["Name"]}')
            logger.info(f'Roles: {role_names}')
            logger.info('-----------------------------------')
            g = Graph(domain = task['Name'], llm_names = llm_names, agent_names = role_names, 
                      decision_method = "FinalRefer", prompt_file = prompt_file, reasoning_name=collab["Name"], **kwargs)
            self.g = g
            final_result.append(g.run(inputs={"query":query}, num_rounds=kwargs["num_rounds"])[0][0])
            costs.append(Cost.instance().value - previous_cost)

        return final_result, costs, log_probs, tasks_probs, vae_loss, agent_num_float
    
    def _preprocess_data(self, raw_data:List[Dict[str, str]]):
        """
        raw_data: List of dictionaries with 'Name' and 'Description' keys
        """
        get_name_description = lambda x: x['Name'] + ' : ' + x['Description']
        return [get_name_description(data) for data in raw_data]
    
    def encoder_roles(self):
        """
        Return:
            task_role_database: Dict[str, List[Dict[str, str]]]: A dictionary of task-role database
            task_role_emb: Dict[str, torch.Tensor]: A dictionary of task-role embeddings. The tensor is N_t_r*d.
        """
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
                        task_role_database[task].append(role_profile)
                        roles_list.append(json.dumps(role_profile))
                if len(roles_list):
                    task_role_emb[task] = self.text_encoder(roles_list).to(self.device)
        logger.info('Role embeddings loaded.')
        return task_role_database, task_role_emb

class TaskClassifier(nn.Module):
    def __init__(self, input_dim:int=384, hidden_dim:int=64, temp:float = 1.0, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.query_encoder = nn.Linear(input_dim, hidden_dim)
        self.task_encoder = nn.Linear(input_dim, hidden_dim)
        self.temp = temp

    def forward(self, queries, tasks):
        """
        queries: N_q*d tensor, N_q is the number of queries, d is the dimension of each query
        tasks: N_t*d tensor, N_t is the number of tasks, d is the dimension of each task
        """ 
        query_embedding = self.query_encoder(queries) # N_q*hidden_dim
        task_embedding = self.task_encoder(tasks) # N_t*hidden_dim
        query_embedding = F.normalize(query_embedding, p=2, dim=1) # L2 normalization
        task_embedding = F.normalize(task_embedding, p=2, dim=1) # L2 normalization
        scores = torch.matmul(query_embedding, task_embedding.T) # N_q*N_t
        scores = F.softmax(scores/self.temp, dim=1) # N_q*N_t
        selected_tasks_id = torch.argmax(scores, dim=1) # N_q

        logger.info(f"Task classification scores: {scores}")

        return selected_tasks_id, scores, query_embedding

class CollabDeterminer(nn.Module):
    def __init__(self, input_dim=384, context_input_dim=384, hidden_dim=64, temp=1.0, device=None):
        super().__init__()
        self.collab_encoder = VAE(input_dim, hidden_dim, hidden_dim)
        self.context_encoder = VAE(context_input_dim, hidden_dim, hidden_dim)
        self.collab_context_encoder = GFusion(d_model=hidden_dim)
        self.temp = temp
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, collabs:torch.Tensor, contexts:torch.Tensor):
        collab_hat, collab_z, collab_mu, collab_logvar = self.collab_encoder(collabs)  # N_t*latent_dim
        collab_z = F.normalize(collab_z, p=2, dim=1) # N_t*latent_dim

        context_hat, context_z, context_mu, context_logvar = self.context_encoder(contexts)  # N_q*latent_dim
        context_z = F.normalize(context_z, p=2, dim=1) # N_q*latent_dim

        scores = torch.matmul(context_z, collab_z.T) # N_q*N_t
        scores = torch.softmax(scores / self.temp, dim=1)

        vae_loss1 = vae_loss_function(collab_hat, collabs, collab_mu, collab_logvar)
        vae_loss2 = vae_loss_function(context_hat, contexts, context_mu, context_logvar)
        vae_loss = vae_loss1 + vae_loss2

        scores_cumsum = torch.cumsum(scores, dim=1)
        random_num = torch.rand([scores.size(0),1], device=self.device)
        selected_index = (scores_cumsum > random_num).float().argmax(dim=1)
        log_probs = torch.log(scores[torch.arange(scores.size(0)), selected_index]).unsqueeze(1)
        collab_embedding = collab_z[selected_index]

        logger.info(f"Collaboration method selection scores: {scores}")
        logger.info(f"Score Collab Mean:{scores.mean(dim=0)}")

        return selected_index, log_probs, collab_embedding, vae_loss


class NumDeterminer(nn.Module):
    def __init__(self, input_dim:int=384, hidden_dim:int = 64, max_agent:int = 6, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae = VAE(input_dim, hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1) # N_q*hidden_dim -> N_q*1
        self.max_agent = max_agent
        
    def forward(self, queries:torch.Tensor):
        """
        num: N_t*input_dim tensor, N_t is the number of reasonings, input_dim is the dimension of each collab
        """ 
        x_hat, z, mu, log_var = self.vae(queries)
        z = F.normalize(z, p=2, dim=1) # L2 normalization

        query_difficulty = self.fc(z) # N_q*1
        query_difficulty = torch.sigmoid(query_difficulty) # N_q*1

        agent_num_float = query_difficulty * self.max_agent # N_q*1
        agent_num_int = torch.clamp(torch.round(agent_num_float), 1, self.max_agent).int() # N_q*1
        vae_loss = vae_loss_function(x_hat, queries, mu, log_var)

        logger.info(f"Number of agents selection scores: {agent_num_float}")
        return agent_num_int, agent_num_float, vae_loss


class RoleAllocation(torch.nn.Module):
    def __init__(self, input_dim:int=384, context_input_dim:int = 128, hidden_dim:int=64, temp=1.0, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_role_embedding = torch.zeros([1, hidden_dim],device=self.device,requires_grad=True) # 1*hidden_dim
        self.role_encoder = VAE(input_dim, hidden_dim, hidden_dim)
        self.context_encoder = nn.Linear(context_input_dim + hidden_dim, hidden_dim) # N_q*(context_input_dim + hidden_dim) -> N_q*hidden_dim
        self.role_context_encoder = GFusion(d_model=hidden_dim) # N_r*hidden_dim, N_q*hidden_dim -> N_r*hidden_dim
        self.temp = temp
        
    def forward(self, roles_list:List[torch.Tensor], contexts:torch.Tensor, agent_num_int:torch.Tensor):
        """
        roles_list: List of roles, each role is a tensor of shape N_r*input_dim, N_r is the number of roles
        contexts: N_q*input_dim tensor, N_q is the number of queries, input_dim is the dimension of each query
        agent_num_int: N_q*1 tensor, N_q is the number of queries, 1 is the number of agents
        """

        selected_roles_idx = [] # List of selected roles List for each query
        log_probs = torch.zeros([contexts.size(0),1], device=self.device) # N_q*1
        summary_role_list = []

        for i, roles in enumerate(roles_list): # each query
            selected_roles_idx.append([])
            role_hat, role_z, role_mu, role_log_var = self.role_encoder(roles) # N_r*hidden_dim
            role_embedding = F.normalize(role_z, p=2, dim=1)

            if i == 0:
                vae_loss = vae_loss_function(role_hat, roles, role_mu, role_log_var)
            else:
                vae_loss = vae_loss_function(role_hat, roles, role_mu, role_log_var) + vae_loss
            current_role_embedding = self.init_role_embedding # 1*hidden_dim
            history_role_embedding = self.init_role_embedding # 1*hidden_dim

            for j in range(agent_num_int[i]): # each agent
                history_role_embedding = history_role_embedding + current_role_embedding
                history_role_embedding = F.layer_norm(history_role_embedding, history_role_embedding.shape[1:])

                contexts_embedding = self.context_encoder(torch.cat([contexts[i].unsqueeze(0), history_role_embedding], dim=1)) # 1*hidden_dim
                contexts_embedding = F.normalize(contexts_embedding, p=2, dim=1) # 1*hidden_dim

                scores = torch.matmul(contexts_embedding, role_embedding.T) # 1*N_r
                scores = torch.softmax(scores/self.temp, dim=1) # 1*N_r
                scores_cumsum = torch.cumsum(scores, dim=1) # N_q*N_t
                random_num = torch.rand([scores.size(0),1], device=self.device) # 1*1
                selected_index = (scores_cumsum > random_num).float().argmax(dim=1) # 1
                log_probs[i][0] = log_probs[i][0] + torch.log(scores[torch.arange(scores.size(0)), selected_index]).unsqueeze(1)

                current_role_embedding = role_embedding[selected_index] # 1*hidden_dim
                selected_roles_idx[-1].append(selected_index)
                logger.info(f"Role selection scores: {scores}")
            summary_role_list.append(history_role_embedding)
            summary_role = torch.cat(summary_role_list, dim=0) # N_q*hidden_dim
        return selected_roles_idx, log_probs, summary_role, vae_loss/len(roles_list)

class LLMRouter(torch.nn.Module):
    def __init__(self, input_dim:int=384, context_input_dim:int = 192, hidden_dim:int=64, temp:float=1.0, max_agent:int=6, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm_encoder = VAE(input_dim, hidden_dim, hidden_dim)
        self.context_encoder = nn.Linear(context_input_dim, hidden_dim) # N_q*context_input_dim -> N_q*hidden_dim
        self.llm_context_encoder = GFusion(d_model=hidden_dim) # N_l*hidden_dim, N_q*hidden_dim -> N_l*hidden_dim
        self.temp = temp
        self.max_agent = max_agent

    def forward(self, llms:torch.Tensor, contexts:torch.Tensor, agent_num_int:torch.Tensor, agent_num_float:torch.Tensor):
        """
        llms: N_l*input_dim tensor, N_l is the number of llms, input_dim is the dimension of each llm
        contexts: N_q*input_dim tensor, N_q is the number of queries, input_dim is the dimension of each query
        """
        llm_hat, llm_z, llm_mu, llm_log_var = self.llm_encoder(llms) # N_l*hidden_dim
        llm_embedding = F.normalize(llm_z, p=2, dim=1) # L2 normalization
        contexts_embedding = self.context_encoder(contexts) # N_q*hidden_dim
        contexts_embedding = F.normalize(contexts_embedding, p=2, dim=1) # L2 normalization
        # llm_context_embedding = self.llm_context_encoder(llm_embedding, contexts_embedding) # N_l*hidden_dim
        # llm_context_embedding = F.normalize(llm_context_embedding, p=2, dim=1) # L2 normalization
        vae_loss = vae_loss_function(llm_hat, llms, llm_mu, llm_log_var)

        scores = torch.matmul(contexts_embedding, llm_embedding.T) # N_q*N_l
        scores = torch.softmax(scores/self.temp, dim=1) # N_q*N_l
        scores_cumsum = torch.cumsum(scores, dim=1)
        selected_llm = torch.zeros([contexts.size(0), llms.size(0)], device=self.device) # N_q*N_l
        selected_llm_index:List[List[int]] = [[] for i in range(contexts.size(0))] # List of selected llm index for each query
        for i in range(1, self.max_agent+1):
            agent_num_mask = (agent_num_int >= i).squeeze(1).float() # N_q
            random_num = torch.rand_like(agent_num_float, device=self.device) # N_q*1
            selected_index = (scores_cumsum > random_num).float().argmax(dim=1) # N_q
            selected_llm[torch.arange(selected_llm.size(0)), selected_index] += agent_num_mask # N_q*N_l

            for j in range(contexts.size(0)):
                if agent_num_mask[j] > 0:
                    selected_llm_index[j].append(int(selected_index[j].item()))
        logger.info(f"LLM selection scores: {scores}")
        logger.info(f"Score LLM Mean:{scores.mean(dim=0)}")
        log_probs = gammaln(agent_num_float + 1) - gammaln(selected_llm + 1).sum(dim=1).unsqueeze(1) + (selected_llm * torch.log(scores)).sum(dim=1).unsqueeze(1) # N_q*1
        
        return selected_llm_index, log_probs, vae_loss
