import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from pypdf import PdfReader

# ----------- CONFIG ------------
PDF_FOLDER = "pdfs"  # Folder containing your PDF files
FAISS_INDEX = "faiss_store"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 3
MODEL = "mistral:7b-instruct-q4"
# -------------------------------


def load_pdfs(folder):
    docs = []
    for file in os.listdir(folder):
        if file.lower().endswith(".pdf"):
            path = os.path.join(folder, file)
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text, metadata={"source": file, "page": i + 1}
                        )
                    )
    return docs


def build_or_load_db(docs):
    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(FAISS_INDEX):
        return FAISS.load_local(
            FAISS_INDEX, embed, allow_dangerous_deserialization=True
        )
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    db = FAISS.from_documents(chunks, embed)
    db.save_local(FAISS_INDEX)
    return db


def main():
    print("📚 Loading PDFs...")
    docs = load_pdfs(PDF_FOLDER)

    print("🔍 Building/Loading FAISS index...")
    db = build_or_load_db(docs)
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})

    print("🤖 Loading Mistral model...")
    llm = Ollama(model=MODEL)

    qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    print("✅ RAG system ready. Type 'exit' to quit.")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = qa({"query": query})
        print("\n--- Answer ---")
        print(result["result"])
        print("\n--- Sources ---")
        for doc in result["source_documents"]:
            print(f"- {doc.metadata['source']} (Page {doc.metadata['page']})")


if __name__ == "__main__":
    main()
