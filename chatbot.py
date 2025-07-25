import streamlit as st
from PyPDF2 import PdfReader

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

# Chunk the text

