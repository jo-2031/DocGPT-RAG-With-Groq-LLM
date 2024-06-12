import os 
import re
import pdfplumber
import bs4
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader

# Load environment variables from .env file
load_dotenv()
 
# Get API key fromt he environmental variables
hugging_face_api_key = os.getenv('HF_token')
groq_api_key = os.getenv('grog_API')



def pdf_to_text_extract(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = []
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=1, y_tolerance=1)
            if page_text:
                all_text.append(page_text)
        return "\n".join(all_text)



#To remove the non alphanumeric values
def text_preprocessing(text):
    lines = text.split('\n')
    alphanumeric_lines = [line for line in lines if re.search(r'\w', line)]
    return "\n".join(alphanumeric_lines) 

# splitting data into chunks
def chunking_data(filtered_text):
    document = Document(page_content=filtered_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap = 64) #chunk_size & chunk_overlap -- need to know!
    splitted_data = text_splitter.split_documents([document])
    return splitted_data


pdf_path = "Attention Is All You Need.pdf"
text = pdf_to_text_extract(pdf_path)
filtered_text = text_preprocessing(text)
splitted_data = chunking_data(filtered_text)
print(filtered_text)
print(text)

# Print the results
for i, chunk in enumerate(splitted_data):
    print(f"Chunk {i+1}:\n{chunk.page_content}\n")

# intializing hugging face embedding - UAE-Large-V1 (Universal AnglE Embedding)
embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hugging_face_api_key, model_name="WhereIsAI/UAE-Large-V1"
)

#words to vectorization and storing to chromaDB (Vector DB)
vectorstore = Chroma.from_documents(splitted_data, embeddings, persist_directory="./db")
vector_retriever = vectorstore.as_retriever(search_kwargs={"k":2})

#integrate the vector_retriever and keyword_retriever (Hybird Search)
keyword_retriever = BM25Retriever.from_documents(splitted_data).k=2
retriever = EnsembleRetriever(retrievers=[vector_retriever,keyword_retriever], weights= [0.5,0.5])





