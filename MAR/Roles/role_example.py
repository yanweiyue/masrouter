{
    "Name": "MathSolver", # str:Role name 
    "MessageAggregation": "PHP", # str: How to aggregate messages from other agents in temporal and spatial dimensions.
    # The order of the list is the priority of the aggregation method.
    # Options: ["Normal", "PythonExecute", "PythonInnerTest", "PHP"]
    "Description": "You are a math expert.\n\
        You will be given a math problem and hints from other agents.\n\
        Give your own solving process step by step based on hints.", # str:Role description about what tasks need to be completed?
    "OutputFormat": "Answer", # str: Output format of the role.
    # Options: ["None", "Text", "Analyze", "Calculation", "Examine", "Answer", "CodeCompletion", "CodeSolver", "Keys"]
    "PostProcess": "Wiki", # str: Post-processing methods for the output.
    # Options: ["None", "PythonExecute", "PythonInnerTest", "Wiki", "Search", "Reflection"]
    "PostDescription": "Reflect on possible errors in the answer above and answer again using the same format.", # str: Post-processing description.
    "PostOutputFormat": "Answer", # str: Post-processing output format.
    # Options: ["None", "Text", "Analyze", "Calculation", "Examine", "Answer", "CodeCompletion", "CodeSolver", "Keys"]
    # None means no post-processing.
}