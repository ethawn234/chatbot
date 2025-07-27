# source 
from PyPDF2 import PdfReader

# common
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# chatbot
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

from dotenv import load_dotenv
load_dotenv()

# Upload PDF files
st.header('My Chatbot')

with st.sidebar:
  st.title('Your documents')
  file = st.file_uploader(' Upload a PDF file and start asking questions', type='pdf')

# Read PDF file
if file is not None:
  pdf_reader = PdfReader(file)
  text = ''
  for page in pdf_reader.pages:
    text += page.extract_text()

  # Chunk the text;
  # Purpose: chunking the text helps Open AI understand the text
  text_splitter = RecursiveCharacterTextSplitter(
    separators='\n',
    chunk_size=1000,
    chunk_overlap=150,  # defines overlap between chunks; eg chunk B may lose context as B may start mid-sentence, and context is contained in a sentence in A.
    length_function=len
  )

  chunk = text_splitter.split_text(text)

  # generate embeddings
  embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
  )

  # create vector store
  vector_store = FAISS.from_texts(chunk, embeddings)

  # get user query
  user_query = st.text_input("What's your question?")

  # do similarity search
  if user_query:
    match = vector_store.similarity_search(user_query)

    # define the LLM
    llm = 

    # output response
    # chain -> take the query, get relevant document, pass to LLM, generate output
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents = match, question = user_query)
    st.write(response)
