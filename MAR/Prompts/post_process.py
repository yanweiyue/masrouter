import re
from typing import Dict
from loguru import logger

from MAR.Tools.coding.python_executor import execute_code_get_return
from MAR.Tools.coding.python_executor import PyExecutor
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Options: ["None", "PythonExecute", "PythonInnerTest", "Wiki", "Search", "Reflection"]
def post_process(raw_inputs:Dict[str,str], output:str, post_method:str):
    if post_method == None or post_method == "None":
        return output
    elif post_method == "PythonExecute":
        return python_execute(raw_inputs, output)
    elif post_method == "PythonInnerTest":
        return python_inner_test(raw_inputs, output)
    elif post_method == "Wiki":
        return wiki(raw_inputs, output)
    elif post_method == "Search":
        return search(raw_inputs, output)
    elif post_method == "Reflection":
        return reflection(raw_inputs, output)
    else:
        raise ValueError(f"Invalid post-processing method: {post_method}")


def python_execute(raw_inputs:Dict[str,str], output:str):
    """
    Execute Python code to post-process the output.
    Args:
        output: str: The output from the LLM.
    Returns:
        Any: str: The post-processed output.
    """
    # Execute Python code to post-process the output
    pattern = r'```python.*```'
    match = re.search(pattern, output, re.DOTALL|re.MULTILINE)
    if match:
        code = match.group(0).lstrip("```python\n").rstrip("\n```")
        output += f"\nthe answer is {execute_code_get_return(code)}"
    return output

def extract_example(prompt: str) -> list:
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
    return results

def python_inner_test(raw_inputs:Dict[str,str], output:str):
    """
    Execute Python code to post-process the output.
    Args:
        raw_inputs: Dict[str,str]: The raw inputs.
        output: str: The output from the LLM.
    Returns:
        Any: str: The post-processed output.
    """
    internal_tests = extract_example(raw_inputs)
    pattern = r'```python.*```'
    match = re.search(pattern, output, re.DOTALL|re.MULTILINE)
    if match:
        code = match.group(0).lstrip("```python\n").rstrip("\n```")
        is_solved, feedback, state = PyExecutor().execute(code, internal_tests, timeout=10)
        if is_solved:
            output += f"\nThe code is solved.\n {feedback}"
        else:
            output += f"\nThe code is not solved.\n {feedback}"
    return output

def wiki(raw_inputs:Dict[str,str], output:str):
    """
    Extract information from Wikipedia to post-process the output.
    Args:
        output: str: The output from the LLM.
    Returns:
        Any: str: The post-processed output.
    """
    # Extract information from Wikipedia to post-process the output
    pattern = r'```keyword.*```'
    match = re.search(pattern, output, re.DOTALL|re.MULTILINE)
    if match:
        keywords = match.group(0).lstrip("```keyword\n").rstrip("\n```")
        # Extract information from Wikipedia
        logger.info(f"keywords: {keywords}")
        keywords = eval(keywords)
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
        for keyword in keywords:
            if type(keyword) == str or type(keyword) == dict:
                output += f"\n{wikipedia.run(keyword)}"
    return output

def search(raw_inputs:Dict[str,str], output:str):
    """
    Search for information to post-process the output.
    Args:
        output: str: The output from the LLM.
    Returns:
        Any: str: The post-processed output.
    """
    #TODO Search for information to post-process the output
    #Maybe bocha or brave search

    return output

def reflection(raw_inputs:Dict[str,str], output:str):
    """
    Reflect on the output to post-process the output.
    Args:
        output: str: The output from the LLM.
    Returns:
        Any: str: The post-processed output.
    """
    return output