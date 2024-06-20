import os
import re
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

# Streamlit app
st.set_page_config(layout="wide", page_title="DocGPT", page_icon="favicon.ico")
st.title("Welcome To DocGPT 🚀")
st.caption("🌟 Retrieval-Augmented Generation (RAG) with LLM Model 🌟")

# Sidebar for API keys and file upload
st.sidebar.title("Configuration")

# Input fields for API keys
hugging_face_token = st.sidebar.text_input("Enter Hugging Face API Key", type="password")
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# Sidebar for file upload
st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Check if both API keys are provided
if hugging_face_token and groq_api_key and uploaded_file is not None:
    # Save the uploaded PDF to a temporary location
    temp_pdf_path = "./temp_uploaded.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Pdf-to-text extraction process
    def pdf_text_extraction(pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            all_text = []
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=1, y_tolerance=1)
                if page_text:
                    all_text.append(page_text)
            return "\n".join(all_text)

    # To remove the non-alphanumeric lines
    def text_preprocessing(text):
        lines = text.split('\n')
        alphanumeric_lines = [line for line in lines if re.search(r'\w', line)]
        return "\n".join(alphanumeric_lines)

    # Splitting the data into chunks
    def chunking_data(filtered_text):
        document = Document(page_content=filtered_text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        splitted_data = text_splitter.split_documents([document])
        return splitted_data

    # Process the PDF
    text = pdf_text_extraction(temp_pdf_path)
    if text:
        filtered_text = text_preprocessing(text)
        splitted_data = chunking_data(filtered_text)

        # Debug information
        st.sidebar.write("Text extraction successful. Number of chunks created: ", len(splitted_data))

        # Initializing hugging face embedding - UAE-Large-V1 (Universal AnglE Embedding)
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=hugging_face_token, model_name="WhereIsAI/UAE-Large-V1"
        )

        # Words to vectorization and storing them in a chromadb (Vector Database)
        vectorstore = Chroma.from_documents(splitted_data, embeddings, persist_directory="./db")
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # Integrating vector_retriever and keyword_retriever (Hybrid Search)
        keyword_retriever = BM25Retriever.from_documents(splitted_data)
        keyword_retriever.k = 2
        retriever = EnsembleRetriever(retrievers=[vector_retriever, keyword_retriever], weights=[0.5, 0.5])

        # Set up ChatGroq
        chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="Llama3-70b-8192")

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

        # Define CSS styles for responsiveness and better design
        st.markdown("""
        <style>
        .chat-placeholder {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #1a1a1a;
            padding: 10px 0;
            box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.3);
        }
        .chat-container {
            display: flex;
            align-items: center;
            padding: 10px;
            margin: 0 auto;
            border-radius: 5px;
            background-color: #262730;
            width: 100%;
            max-width: 800px;
        }
        .user-msg {
            justify-content: flex-start;
        }
        .assistant-msg {
            justify-content: flex-start;
        }
        .chat-container img {
            width: 30px;
            height: 30px;
            margin-right: 10px;
        }
        .chat-container span {
            color: white;
            margin: 0 10px;
        }
        @media (max-width: 600px) {
            .chat-container {
                width: 90%;
                flex-direction: column;
                align-items: flex-start;
            }
            .chat-container img {
                width: 20px;
                height: 20px;
                margin-right: 5px;
            }
            .chat-container span {
                margin: 5px 0;
            }
        }
        </style>
        """, unsafe_allow_html=True)

        # Chat container
        chat_placeholder = st.container()
        with chat_placeholder:
            for message in st.session_state.conversation:
                if message.startswith("User:"):
                    user_query = message[6:]
                    st.markdown(
                        f'<div class="chat-container user-msg">'
                        f'<img src="https://img.icons8.com/color/48/000000/user-male-circle--v2.png">'
                        f'<span>{user_query}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    assistant_response = message[10:]
                    st.markdown(
                        f'<div class="chat-container assistant-msg">'
                         f'<img src="https://img.icons8.com/color/48/000000/bot.png">'
                        f'<span>{assistant_response}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # Setup for the conversation
        query = st.text_input("", key="query_input", placeholder="Type your question here...")

        if st.button("Send") and query:
            st.session_state.conversation.append(f"User: {query}")

            response = ""
            for chunk in chain.stream(query):
                response += chunk

            st.session_state.conversation.append(f"Assistant: {response}")
            st.experimental_rerun()
    else:
        st.sidebar.write("Failed to extract text from the PDF.")
else:
    st.write("Please provide both API keys and upload a PDF file to start the conversation.")
