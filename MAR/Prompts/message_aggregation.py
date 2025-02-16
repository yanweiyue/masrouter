import re
from typing import Dict

from MAR.Tools.coding.python_executor import execute_code_get_return
from MAR.Tools.coding.python_executor import PyExecutor
from Datasets.gsm8k_dataset import gsm_get_predict

# ["Normal", "PythonExecute", "PythonInnerTest", "PHP"]
def message_aggregation(raw_inputs:Dict[str,str], messages:Dict[str,Dict], aggregation_method):
    """
    Aggregate messages from other agents in temporal and spatial dimensions.
    Args:
        messages: Dict[str,Dict]: A dict of messages from other agents.
        aggregation_method: str: Aggregation method.
    Returns:
        Any: str: Aggregated message.
    """
    if aggregation_method == "Normal":
        return normal_agg(raw_inputs, messages)
    elif aggregation_method == "PythonExecute":
        return python_execute(raw_inputs, messages)
    elif aggregation_method == "PythonInnerTest":
        return python_inner_test(raw_inputs, messages)
    elif aggregation_method == "PHP":
        return php(raw_inputs, messages)
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation_method}")

def normal_agg(raw_inputs:Dict[str,str], messages:Dict[str,Dict]):
    """
    Aggregate messages from other agents in temporal and spatial dimensions normally.
    Args:
        messages: Dict[str,Dict]: A dict of messages from other agents.
    Returns:
        Any:str:Aggregated message.
    """
    # Aggregate messages normally
    aggregated_message = ""
    for id, info in messages.items():
        aggregated_message += f"Agent {id}, role is {info['role'].role}, output is:\n\n {info['output']}\n\n"
    return aggregated_message

def python_execute(raw_inputs:Dict[str,str],messages:Dict[str,Dict]):
    """
    Aggregate messages from other agents in temporal and spatial dimensions by executing Python code.
    Args:
        messages: Dict[str,Dict]: A dict of messages from other agents.
    Returns:
        Any:str: Aggregated message.
    """
    # Execute Python code to aggregate messages
    aggregated_message = ""
    hints = "(Hint: The answer is near to"
    pattern = r'```python.*```'
    for id, info in messages.items():
        aggregated_message += f"Agent {id}, role is {info['role'].role}, output is:\n\n {info['output']}\n\n"
        match = re.search(pattern, info['output'], re.DOTALL|re.MULTILINE)
        if match:
            code = match.group(0).lstrip("```python\n").rstrip("\n```") # the result must in the local vars answer
            answer = execute_code_get_return(code)
            if answer:
                hints += f" {answer},"
                aggregated_message += f"The execution result of the code is {answer}."
    hints += ")."
    aggregated_message += hints if hints != "(Hint: The answer is near to)." else ""
    return aggregated_message

def extract_example(prompt: Dict[str,str]) -> list:
    # the prompt['query'] only contains the code snippet
    prompt = prompt['query']
    lines = (line.strip() for line in prompt.split('\n') if line.strip())
    results = []
    lines_iter = iter(lines)
    for line in lines_iter:
        if line.startswith('>>>'):
            function_call = line[4:]
            expected_output = next(lines_iter, None)
            if expected_output:
                results.append(f"assert {function_call} == {expected_output}")
        if line.startswith('assert'):
            results.append(line)
    return results

def python_inner_test(raw_inputs:Dict[str,str],messages:Dict[str,Dict]):
    """
    Aggregate messages from other agents in temporal and spatial dimensions by running inner tests.
    Args:
        messages: Dict[str,Dict]: A dict of messages from other agents.
    Returns:
        Any:str: Aggregated message.
    """
    # Run inner tests to aggregate messages
    internal_tests = extract_example(raw_inputs)
    aggregated_message = ""
    pattern = r'```python.*```'
    for id, info in messages.items():
        aggregated_message += f"Agent {id}, role is {info['role'].role}, output is:\n\n {info['output']}\n\n"
        match = re.search(pattern, info['output'], re.DOTALL|re.MULTILINE)
        if match:
            code = match.group(0).lstrip("```python\n").rstrip("\n```")
            is_solved, feedback, state = PyExecutor().execute(code, internal_tests, timeout=100)
            if is_solved:
                aggregated_message += f"\nThe code is solved.\n {feedback}"
            else:
                aggregated_message += f"The code is not solved.\n {feedback}"
    return aggregated_message

def php(raw_inputs:Dict[str,str],messages:Dict[str,Dict]):
    """
    Aggregate messages from other agents in temporal and spatial dimensions using PHP.
    Args:
        messages: Dict[str,Dict]: A dict of messages from other agents.
    Returns:
        Any:str: Aggregated message.
    """
    # Use PHP to aggregate messages
    aggregated_message = ""
    hints = "(Hint: The answer is near to"
    python_pattern = r'```python.*```'
    
    for id, info in messages.items():
        aggregated_message += f"Agent {id}, role is {info['role'].role}, output is:\n\n {info['output']}\n\n"
        python_match = re.search(python_pattern, info['output'], re.DOTALL|re.MULTILINE)
        if python_match:
            code = python_match.group(0).lstrip("```python\n").rstrip("\n```") # the result must in the local vars answer
            answer = execute_code_get_return(code)
            if answer:
                hints += f" {answer},"
                aggregated_message += f"The execution result of the code is {answer}."
        if 'the answer is ' in info['output'] or 'The answer is ' in info['output']:
            answer = gsm_get_predict(info['output'])
            hints += f" {answer},"
    hints += ")."
    aggregated_message += hints if hints != "(Hint: The answer is near to)." else ""
    return aggregated_message

def inner_test(raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], ):
    # Use inter tests to aggregate messages
    internal_tests = extract_example(raw_inputs)
    if internal_tests == []:
        return False, ""
    pattern = r'```python.*```'
    for id, info in spatial_info.items():
        match = re.search(pattern, info['output'], re.DOTALL|re.MULTILINE)
        if match:
            code = match.group(0).lstrip("```python\n").rstrip("\n```")
            is_solved, feedback, state = PyExecutor().execute(code, internal_tests, timeout=10)
            if is_solved:
                return is_solved, info['output']
    for id, info in temporal_info.items():
        match = re.search(pattern, info['output'], re.DOTALL|re.MULTILINE)
        if match:
            code = match.group(0).lstrip("```python\n").rstrip("\n```")
            is_solved, feedback, state = PyExecutor().execute(code, internal_tests, timeout=10)
            if is_solved:
                return is_solved, info['output']
    return False, ""
    