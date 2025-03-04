from apikeys import init_keys
init_keys()

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools import tool
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA

docs_location = os.path.join(os.curdir, "docs")

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
splitter = RecursiveCharacterTextSplitter(separators=['\n'], chunk_size=1000)

# print("Uploaded documents:")
# for doc in os.listdir(docs_location):
#     print(doc)
# print("\n\n")

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
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6, "k": 3})

    return retriever


retriever = get_retriever(docs_location, splitter, embeddings)

docs = retriever.invoke("What are the 5 most common ML algorithms?")

