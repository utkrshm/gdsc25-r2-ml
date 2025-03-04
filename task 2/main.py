import os
import random
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Initializing all API Keys [Google, Tavily]
os.environ["GOOGLE_API_KEY"] = "YOUR-GOOGLE-API-KEY"
os.environ["TAVILY_API_KEY"] = "YOUR-TAVILY-API-KEY"

# Creating the base things needed for RAG
docs_location = os.path.join(os.curdir, "docs")
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=os.environ['GOOGLE_API_KEY'])
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["GOOGLE_API_KEY"])
splitter = RecursiveCharacterTextSplitter(separators='\n', chunk_size=1000)

def get_thread_id():
    thread_id = ""
    
    for _ in range(3): thread_id += random.choice("abcdefghijklmnopqrstuvwxyz")
    for _ in range(3): thread_id += random.choice("1234567890")
    
    return thread_id

# Creating a function to get vectorstore retriever
def get_retriever(docs_location: os.PathLike, text_splitter: RecursiveCharacterTextSplitter, embedding: GoogleGenerativeAIEmbeddings):
    all_docs = []
    
    for doc_path in os.listdir(docs_location):
        doc_path = docs_location + "\\" + doc_path
        
        extension = doc_path.split(".")[-1].lower()
        
        if extension == 'pdf':  loader = PyPDFLoader(doc_path)
        elif extension == 'txt':  loader = TextLoader(doc_path)
        
        loaded_doc = loader.load()
        for doc in loaded_doc:
            doc.metadata = {"name": doc_path.split("/")[-1], }
        
        all_docs.extend(loaded_doc)
        
    chunks = text_splitter.split_documents(all_docs)
    vector_store = FAISS.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3, "k": 3})

    return retriever

# Getting the retriever
retriever = get_retriever(docs_location, splitter, embeddings)

# To get information about the data that the user has uploaded in the knowledge base
# docs_info = input("What is the information in the documents that have been uploaded?: ")
docs_info = "Info about me, the lab experiments in my Physics Lab this semester, the tasks of GDSC-VIT Enrollments '25 Round 2, and a copy of the book \"50 most asked ML Q&As\""

# Create retriever tool
retrieval_tool_desc = f"""
This is a retrieval tool for when you need context about something the user has asked and you don't have an answer. 
Here's the information that the user says they have fed into the external knowledge base: {docs_info} 
"""
retrieval_tool = create_retriever_tool(retriever, name='context_retrieval_tool', description=docs_info) 


# Creating tool for Arithmetic Input
class ArithmeticInput(BaseModel):
    operation: str = Field(
        description="The operation that needs to be performed.",
        examples=["2+3", "(3+4)-(2+4)"]
    )

@tool(args_schema=ArithmeticInput)
def arithmetic(operation: str):
    """Performs arithmetic operation
    
    Args:
        operation (str): The operation that needs to be performed.
            Example. If 2 and 3 are to be added, then operation is "2+3"
            If 2 and 4 are to be subtracted from the sum of 3 + 4, then operation is "(3+4)-(2+4)"
    """
    try:
        return eval(operation)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Creating tool for Searching
search_tool = TavilySearchResults(max_results=2, include_images=False, name='tavily_search')

tools = [search_tool, arithmetic, retrieval_tool]
memory = MemorySaver()

agent_executor = create_react_agent(model=llm, tools=tools, checkpointer=memory)

thread_id = get_thread_id()

config = {"configurable": {"thread_id": thread_id}}

print("\n\nThread ID: " + thread_id)

print("Entering chat mode with AI agent.... Please type 'quit' or 'exit' to exit")
while True:
    user_input = input("Human: ")
    
    if user_input == 'quit'.casefold() or user_input == "exit".casefold():
        print("Exiting loop... Thank you for trying out the Agent")
        break
    
    history = agent_executor.invoke(input={"messages": HumanMessage(content=user_input)}, config=config)
    print("AI: " + history['messages'][-1].content)
