from apikeys import init_keys

import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Initializing all API Keys [Google, Tavily, Pinecone]
init_keys()

class ArithmeticInput(BaseModel):
    operation: str = Field(
        description="The operation that needs to be performed.",
        examples=["2+3", "(3+4)-(2+4)"]
    )

# Use the schema with your tool
@tool(args_schema=ArithmeticInput)
def arithmetic(operation: str):
    """Performs arithmetic operation
    
    Args:
        operation (str): The operation that needs to be performed.
            Example. If 2 and 3 are to be added, then operation is "2+3"
            If 2 and 4 are to be subtracted from the sum of 3 + 4, then operation is "(3+4)-(2+4)"
    """
    try:
        # For more security, you could replace eval with a proper parser
        return eval(operation)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


search = TavilySearchResults(max_results=2, include_images=False)

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=os.environ['GOOGLE_API_KEY'])
llm_with_tools = llm.bind_tools([arithmetic, search])

print(llm_with_tools.invoke(input="Who won the last cricket match between India and New Zealand?").tool_calls)
