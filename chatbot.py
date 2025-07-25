import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    chunk_overlap=150, # defines overlap between chunks; eg chunk B may lose context as B may start mid-sentence, and context is contained in a sentence in A.
    length_function=len
  )

  chunks = text_splitter.split_text(text)
  st.write(chunks)
