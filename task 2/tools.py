from langchain.tools import tool

@tool
def calculate(operation: str):
    "Make an arithmetic calculation"
    return eval(operation)