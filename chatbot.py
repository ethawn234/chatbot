import streamlit as st

# Upload PDF files
st.header("My Chatbot")

with st.sidebar:
  st.title("Your documents")
  file = st.file_uploader(" Upload a PDF file and start asking questions", type="pdf")
