from typing import Dict
import json
from loguru import logger

from MAR.Agent.agent_registry import AgentRegistry
from MAR.LLM.llm_registry import LLMRegistry
from MAR.Roles.role_registry import RoleRegistry
from MAR.Graph.node import Node
from MAR.Prompts.message_aggregation import message_aggregation,inner_test
from MAR.Prompts.post_process import post_process
from MAR.Prompts.output_format import output_format_prompt
from MAR.Prompts.reasoning import reasoning_prompt


@AgentRegistry.register('Agent')
class Agent(Node):
    def __init__(self, id: str | None =None, domain: str = "", role:str = None , llm_name: str = "",reason_name: str = "",):
        super().__init__(id, reason_name, domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.role = RoleRegistry(domain, role)
        self.reason = reason_name

        self.message_aggregation = self.role.get_message_aggregation()
        self.description = self.role.get_description()
        self.output_format = self.role.get_output_format()
        self.post_process = self.role.get_post_process()
        self.post_description = self.role.get_post_description()
        self.post_output_format = self.role.get_post_output_format()
        # Reflect
        if reason_name == "Reflection" and self.post_output_format == "None":
            self.post_output_format = self.output_format
            self.post_description = "\nReflect on possible errors in the answer above and answer again using the same format. If you think there are no errors in your previous answers that will affect the results, there is no need to correct them.\n"
    
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str, Dict], temporal_info:Dict[str, Dict], **kwargs):
        query = raw_inputs['query']
        spatial_prompt = message_aggregation(raw_inputs, spatial_info, self.message_aggregation)
        temporal_prompt = message_aggregation(raw_inputs, temporal_info, self.message_aggregation)
        format_prompt = output_format_prompt[self.output_format]
        reason_prompt = reasoning_prompt[self.reason]

        system_prompt = f"{self.description}\n{reason_prompt}"
        system_prompt += f"\nFormat requirements that must be followed:\n{format_prompt}" if format_prompt else ""
        user_prompt = f"{query}\n"
        user_prompt += f"At the same time, other agents' outputs are as follows:\n\n{spatial_prompt}" if spatial_prompt else ""
        user_prompt += f"\n\nIn the last round of dialogue, other agents' outputs were:\n\n{temporal_prompt}" if temporal_prompt else ""
        return [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict],**kwargs):
        """
        Run the agent.
        Args:
            inputs: dict[str, str]: Raw inputs.
            spatial_info: dict[str, dict]: Spatial information.
            temporal_info: dict[str, dict]: Temporal information.
        Returns:
            Any: str: Aggregated message.
        """
        query = input['query']
        passed, response= inner_test(input, spatial_info, temporal_info)
        if passed:
            return response
        prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        response = self.llm.gen(prompt)
        response = post_process(input, response, self.post_process)
        logger.debug(f"Agent {self.id} Role: {self.role.role} LLM: {self.llm.model_name}")
        logger.debug(f"system prompt:\n {prompt[0]['content']}")
        logger.debug(f"user prompt:\n {prompt[1]['content']}")
        logger.debug(f"response:\n {response}")

        # #! 
        # received_id = []
        # for id, info in spatial_info.items():
        #     role = info["role"].role
        #     received_id.append(id + '(' + role + ')')
        # for id, info in temporal_info.items():
        #     role = info["role"].role
        #     received_id.append(id + '(' + role + ')')

        # entry = {
        #     "id": self.id,
        #     "role": self.role.role,
        #     "llm_name": self.llm.model_name,
        #     "system_prompt": prompt[0]['content'],
        #     "user_prompt": prompt[1]['content'],
        #     "received_id": received_id,
        #     "response": response,
        # }
        # try:
        #     with open(f'./result/tmp_log.json', 'r', encoding='utf-8') as f:
        #         data = json.load(f)
        # except (FileNotFoundError, json.JSONDecodeError):
        #     data = []

        # data.append(entry)

        # with open(f'./result/tmp_log.json', 'w', encoding='utf-8') as f:
        #     json.dump(data, f, ensure_ascii=False, indent=2)
        # #!

        post_format_prompt = output_format_prompt[self.post_output_format]
        if post_format_prompt is not None:
            system_prompt = f"{self.post_description}\n"
            system_prompt += f"Format requirements that must be followed:\n{post_format_prompt}"
            user_prompt = f"{query}\nThe initial thinking information is:\n{response} \n Please refer to the new format requirements when replying."
            prompt = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
            response = self.llm.gen(prompt)
            logger.debug(f"post system prompt:\n {system_prompt}")
            logger.debug(f"post user prompt:\n {user_prompt}")
            logger.debug(f"post response:\n {response}")
            
            # #! 
            # received_id = []
            # role = self.role.role
            # received_id.append(self.id + '(' + role + ')')

            # entry = {
            #     "id": self.id,
            #     "role": self.role.role,
            #     "llm_name": self.llm.model_name,
            #     "system_prompt": prompt[0]['content'],
            #     "user_prompt": prompt[1]['content'],
            #     "received_id": received_id,
            #     "response": response,
            # }
            # try:
            #     with open(f'./result/tmp_log.json', 'r', encoding='utf-8') as f:
            #         data = json.load(f)
            # except (FileNotFoundError, json.JSONDecodeError):
            #     data = []

            # data.append(entry)

            # with open(f'./result/tmp_log.json', 'w', encoding='utf-8') as f:
            #     json.dump(data, f, ensure_ascii=False, indent=2)
            # #!
        return response
    
    def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        return None

@AgentRegistry.register('FinalRefer')
class FinalRefer(Node):
    def __init__(self, id: str | None =None, agent_name = "", domain = "", llm_name = "", prompt_file = ""):
        super().__init__(id, agent_name, domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_file = json.load(open(f"{prompt_file}", 'r', encoding='utf-8'))

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):  
        system_prompt = f"{self.prompt_file['system']}"
        spatial_str = ""
        for id, info in spatial_info.items():
            spatial_str += id + ": " + info['output'] + "\n\n"
        user_prompt = f"The task is:\n\n {raw_inputs['query']}.\n At the same time, the output of other agents is as follows:\n\n{spatial_str} {self.prompt_file['user']}"
        return [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
    
    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        response = self.llm.gen(prompt)
        logger.debug(f"Final Refer Node LLM: {self.llm.model_name}")
        logger.debug(f"Final System Prompt:\n {prompt[0]['content']}")
        logger.debug(f"Final User Prompt:\n {prompt[1]['content']}")
        logger.debug(f"Final Response:\n {response}")
        # #! 
        # received_id = []
        # for id, info in spatial_info.items():
        #     role = info["role"].role
        #     received_id.append(id + '(' + role + ')')
        # for id, info in temporal_info.items():
        #     role = info["role"].role
        #     received_id.append(id + '(' + role + ')')

        # entry = {
        #     "id": self.id,
        #     "role": "FinalDecision",
        #     "llm_name": self.llm.model_name,
        #     "system_prompt": prompt[0]['content'],
        #     "user_prompt": prompt[1]['content'],
        #     "received_id": received_id,
        #     "response": response,
        # }
        # try:
        #     with open(f'./result/tmp_log.json', 'r', encoding='utf-8') as f:
        #         data = json.load(f)
        # except (FileNotFoundError, json.JSONDecodeError):
        #     data = []

        # data.append(entry)

        # with open(f'./result/tmp_log.json', 'w', encoding='utf-8') as f:
        #     json.dump(data, f, ensure_ascii=False, indent=2)
        # #!
        return response
    
    def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        return None