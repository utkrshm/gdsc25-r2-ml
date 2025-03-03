from apikeys import init_keys

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Initializing all API Keys
init_keys()

@tool
def arithmetic(operation):
    """Performs arithmetic operation

    Args:
        operation (str): The operation that needs to be performed.
            Example. If 2 and 3 are to be added, then operation is "2+3"
                     If 2 and 4 are to be subtracted from the sum of 3 + 4, then operation is "(3+4)-(2+4)"
    """
    return eval(operation) 

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=os.environ['GOOGLE_API_KEY'])
print(llm.invoke("What is the capital of India?"))