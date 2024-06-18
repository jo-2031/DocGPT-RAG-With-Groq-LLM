import os
import re
import pdfplumber
import streamlit as st
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

# Get API key from the environmental variables
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

# To remove the non-alphanumeric values
def text_preprocessing(text):
    lines = text.split('\n')
    alphanumeric_lines = [line for line in lines if re.search(r'\w', line)]
    return "\n".join(alphanumeric_lines) 

# Splitting data into chunks
def chunking_data(filtered_text):
    document = Document(page_content=filtered_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    splitted_data = text_splitter.split_documents([document])
    return splitted_data

# Streamlit app
st.title("PDF Chatbot with LangChain")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded PDF to a temporary location
    temp_pdf_path = "./temp_uploaded.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Process the PDF
    text = pdf_to_text_extract(temp_pdf_path)
    filtered_text = text_preprocessing(text)
    splitted_data = chunking_data(filtered_text)

    # Initialize Hugging Face embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hugging_face_api_key, model_name="WhereIsAI/UAE-Large-V1"
    )

    # Create vector store and retriever
    vectorstore = Chroma.from_documents(splitted_data, embeddings, persist_directory="./db2")
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k":2})

    # Integrate vector retriever and keyword retriever (Hybrid Search)
    keyword_retriever = BM25Retriever.from_documents(splitted_data)
    keyword_retriever.k = 2
    retriever = EnsembleRetriever(retrievers=[vector_retriever, keyword_retriever], weights=[0.5, 0.5])

    # Set up ChatGroq
    chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

    template = """
    User: You are an AI Assistant that follows instructions extremely well.
    Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

    Keep in mind, you will lose the job, if you answer out of CONTEXT questions

    CONTEXT: {context}
    Query: {question}

    Remember only return AI answer
    Assistant:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    chain = (
        {
            "context": retriever.with_config(run_name="Docs"),
            "question": RunnablePassthrough(),
        }
        | prompt
        | chat
        | output_parser
    )

 # Initialize session state for conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    query = st.text_input("Ask a question about the PDF content")

    if query:
        # Add user query to conversation history
        st.session_state.conversation.append(f"User: {query}")

        # Get AI response
        response = ""
        for chunk in chain.stream(query):
            response += chunk

        # Add AI response to conversation history
        st.session_state.conversation.append(f"Assistant: {response}")

        # Clear the input box
        st.text_input("Ask a question about the PDF content", value="", key="query_input")

    # Display conversation history
    if st.session_state.conversation:
        st.write("\n".join(st.session_state.conversation))
