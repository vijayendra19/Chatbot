import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG Chatbot", page_icon=":robot_face:")
st.title("RAG Chatbot")
st.write("Ask questions about C++ programming ")

@st.cache_resource
def load_vector_store():
    # Load documents
    loader = TextLoader("C++_Introduction.txt",encoding="utf-8")
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    final_documents = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(final_documents, embeddings)

    return db


db=load_vector_store()

query = st.text_input("Enter your question about C++ programming:")
if query:
    docs=db.similarity_search(query,k=3)
    st.subheader("Relevant Context:")
    for i, doc in enumerate(docs):
        st.markdown(f"**Result {i+1}:**")
        st.write(doc.page_content)
        
    