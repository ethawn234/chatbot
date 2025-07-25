import os

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

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
    # st.write(text)

  # Chunk the text;
  # Purpose: chunking the text helps Open AI understand the text
  text_splitter = RecursiveCharacterTextSplitter(
    separators='\n',
    chunk_size=1000,
    chunk_overlap=150,  # defines overlap between chunks; eg chunk B may lose context as B may start mid-sentence, and context is contained in a sentence in A.
    length_function=len
  )

  chunk = text_splitter.split_text(text)
  # st.write(chunks)

  # generate embeddings
  embeddings = OpenAIEmbeddings(api_key=api_key)  # configure api key usage via .env

  # create vector store (FAISS: Facebook Semantic Search)
  # FAISS is doing 3 things here:
  # 1. generates embeddings
  # 2. initializes FAISS
  # 3. stores the chunks & embeddings
  vector_store = FAISS.from_texts(chunk, embeddings)
