import streamlit as st
import pandas as pd
import os
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

st.set_page_config(page_title="Medical Assistant RAG", page_icon="🩺", layout="wide")

st.title("🩺 Medical Transcription AI Assistant")
st.markdown("Ask questions based on the medical transcription database.")

with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    st.markdown("[Get a Groq API Key](https://console.groq.com/keys)")
    
    st.divider()
    st.markdown("### Model Details")
    st.markdown("- **LLM:** Llama-3.3-70b-versatile")
    st.markdown("- **Embeddings:** all-MiniLM-L6-v2")
    st.markdown("- **Data:** Medical Transcriptions")


@st.cache_resource
def load_vector_db():
    """
    Loads the vector database. 
    1. Tries to load a local FAISS index.
    2. If not found, processes the CSV and creates a new index.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_path = "faiss_index"

    # Check if pre-built index exists
    if os.path.exists(faiss_path):
        st.toast("Loading existing vector store...", icon="💾")
        return FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
    
    # Otherwise build from CSV
    if not os.path.exists("mtsamples.csv"):
        st.error("mtsamples.csv not found! Please place the dataset in the app directory.")
        return None

    with st.spinner("Building Knowledge Base (this takes a minute)..."):
        # Load and clean data
        df = pd.read_csv('mtsamples.csv')
        df = df.dropna(subset=['transcription'])
        df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')
        df['description'] = df['description'].fillna('')
        
        # Create context text
        df['full_text'] = (
            "Specialty: " + df['medical_specialty'] + "\n" +
            "Case: " + df['description'] + "\n" +
            "Content: " + df['transcription']
        )

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = []
        for _, row in df.iterrows():
            chunks = splitter.split_text(row['full_text'])
            for chunk in chunks:
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        'specialty': row['medical_specialty'],
                        'description': row['description'][:100]
                    }
                ))

        # Create Vector Store
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(faiss_path)
        st.toast("Index built and saved!", icon="✅")
        
        return vectorstore

# Initialize Vector Store
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vector_db()

# --- 2. Chat Logic ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ex: What are the symptoms of diabetes?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check dependencies
    if not groq_api_key:
        st.error("Please enter your Groq API Key in the sidebar to proceed.")
        st.stop()
        
    if st.session_state.vectorstore is None:
        st.error("Vector Store not initialized. Please check dataset.")
        st.stop()

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Initialize LLM
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=512
            )

            # Setup Retriever
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

            # Setup Prompt
            template = """Use the medical context to answer the question accurately.
            Cite the medical specialty when relevant.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context: {context}

            Question: {question}

            Answer:"""
            
            QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )

            # Run Chain
            response = qa_chain.invoke({"query": prompt})
            result_text = response['result']
            source_docs = response['source_documents']

            # Display Result
            message_placeholder.markdown(result_text)
            
            # Display Sources in Expander
            with st.expander("📚 View Medical Sources"):
                for i, doc in enumerate(source_docs):
                    st.markdown(f"**Source {i+1} (Specialty: {doc.metadata['specialty']})**")
                    st.info(doc.page_content)

            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": result_text})

        except Exception as e:

            st.error(f"An error occurred: {str(e)}")

