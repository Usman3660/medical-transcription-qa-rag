import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd

# Page config
st.set_page_config(
    page_title="Medical RAG QA System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_rag_pipeline():
    """Load the RAG pipeline with caching"""
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets["GROQ_API_KEY"]
        os.environ["GROQ_API_KEY"] = api_key
        
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load FAISS vector store
        vectorstore = FAISS.load_local(
            "medical_faiss",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Initialize LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=512
        )
        
        # Create prompt template
        template = """You are a medical assistant. Use the medical context below to answer the question accurately.
Cite the medical specialty when relevant. If you don't know, say so.

Context: {context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Error loading pipeline: {str(e)}")
        return None

# Header
st.markdown('<h1 class="main-header">🏥 Medical RAG QA System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This system uses Retrieval-Augmented Generation (RAG) to answer medical questions 
    based on 4,999 clinical transcriptions.
    
    **Features:**
    - 🔍 Semantic search across medical records
    - 🤖 AI-powered answer generation
    - 📚 Source citation and grounding
    - ⚡ Fast retrieval with FAISS
    """)
    
    st.header("📊 Stats")
    st.metric("Medical Records", "4,999")
    st.metric("Text Chunks", "~15,000")
    st.metric("Medical Specialties", "40+")
    
    st.header("🔧 Configuration")
    st.code("""
Model: Llama 3.3 70B
Embeddings: all-MiniLM-L6-v2
Vector DB: FAISS
Top-k: 3 sources
    """)
    
    st.header("⚠️ Disclaimer")
    st.warning("This is for educational purposes only. Not for clinical use.")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask a Medical Question")
    
    # Example questions
    examples = [
        "What are the symptoms of diabetes?",
        "How is hypertension diagnosed?",
        "What is the treatment for pneumonia?",
        "Describe the procedure for colonoscopy",
        "What are complications of coronary artery disease?"
    ]
    
    selected_example = st.selectbox(
        "Or choose an example:",
        [""] + examples,
        index=0
    )
    
    # Question input
    question = st.text_area(
        "Enter your question:",
        value=selected_example if selected_example else "",
        height=100,
        placeholder="e.g., What are the risk factors for stroke?"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        submit = st.button("🔍 Get Answer", type="primary")
    with col_btn2:
        clear = st.button("🗑️ Clear")
    
    if clear:
        st.session_state.history = []
        st.rerun()

with col2:
    st.subheader("📈 Quick Stats")
    if st.session_state.history:
        st.metric("Questions Asked", len(st.session_state.history))
        avg_sources = sum(h['num_sources'] for h in st.session_state.history) / len(st.session_state.history)
        st.metric("Avg Sources/Query", f"{avg_sources:.1f}")

# Process question
if submit and question:
    with st.spinner("🔍 Searching medical records and generating answer..."):
        # Load pipeline
        if st.session_state.qa_chain is None:
            st.session_state.qa_chain = load_rag_pipeline()
        
        if st.session_state.qa_chain:
            try:
                # Get answer
                result = st.session_state.qa_chain({"query": question})
                
                # Display answer
                st.success("✅ Answer Generated")
                st.markdown("### 📝 Answer")
                st.markdown(f"<div style='background-color: #e8f4f8; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #1f77b4;'>{result['result']}</div>", unsafe_allow_html=True)
                
                # Display sources
                st.markdown("### 📚 Source Documents")
                sources = result['source_documents']
                
                for i, doc in enumerate(sources, 1):
                    with st.expander(f"📄 Source {i}: {doc.metadata.get('specialty', 'Unknown')}"):
                        st.markdown(f"**Medical Specialty:** {doc.metadata.get('specialty', 'N/A')}")
                        st.markdown(f"**Description:** {doc.metadata.get('description', 'N/A')}")
                        st.markdown("**Content:**")
                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                
                # Add to history
                st.session_state.history.append({
                    'question': question,
                    'answer': result['result'],
                    'num_sources': len(sources)
                })
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
        else:
            st.error("Failed to load RAG pipeline. Check your configuration.")

# History section
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Query History")
    
    for i, item in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"Q{len(st.session_state.history) - i + 1}: {item['question'][:60]}..."):
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Answer:** {item['answer']}")
            st.caption(f"Sources: {item['num_sources']}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with LangChain • Groq • FAISS • Streamlit</p>
    <p>📊 <a href='https://github.com/YOUR_USERNAME/medical-rag-qa-system'>View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
