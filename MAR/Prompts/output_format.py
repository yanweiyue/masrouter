# Options: ["None", "Text", "Analyze", "Calculation", "Examine", "Answer", "CodeCompletion", "CodeSolver", "Keys"]
output_format_prompt = {
    "None": None,
    "Text": "",
    "Calculation": "Please provide the formula for the problem and bring in the numerical values to solve the problem.\n\
The last line of your output must contain only the final result without any units or redundant explanation,\
for example: The answer is 140\n\
If it is a multiple choice question, please output the options. For example: The answer is A.\n\
However, The answer is 140$ or The answer is Option A or The answer is A.140 is not allowed.",
    "Examine": "If you are provided with other responses, check that they are correct and match each other.\n\
Check whether the logic/calculation of the problem solving and analysis process is correct(if present).\n\
Check whether the code corresponds to the solution analysis(if present).\n\
Give your own complete solving process using the same format." ,
    "Answer": "The last line of your output must contain only the final result without any units or redundant explanation,\
for example: The answer is 140\n\
If it is a multiple choice question, please output the options. For example: The answer is A.\n\
However, The answer is 140$ or The answer is Option A or The answer is A.140 is not allowed.\n",
    "CodeCompletion": "You will be given a function signature and its docstring by the user.\n\
Write your full implementation of this function.\n\
Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```\n\
Do not change function names and input variable types in tasks.",
    "CodeSolver": "Analyze the question and write functions to solve the problem.\n\
The function should not take any arguments and use the final result as the return value.\n\
The last line of code calls the function you wrote and assigns the return value to the \(answer\) variable.\n\
Use a Python code block to write your response. For example:\n```python\ndef fun():\n x = 10\n y = 20\n return x + y\nanswer = fun()\n```\n",
    "Keys": "Please provide relevant keywords that need to be searched on the Internet, relevant databases, or Wikipedia.\n\
Use a key word block to give a list of keywords of your choice.\n\
Please give a few concise short keywords, the number should be less than four\n\
For example:\n```keyword\n['catfish effect', 'Shakespeare', 'global warming']\n```\n\
If there is no entity in the question that needs to be searched, you don't have to provide it."
                       }