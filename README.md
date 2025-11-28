# medical-transcription-qa-rag
🩺 Medical Transcription AI Assistant

A Retrieval-Augmented Generation (RAG) application that acts as an intelligent medical assistant. It allows users to ask questions about medical cases and retrieves accurate answers based on a database of medical transcriptions.

🔗 Live Demo: https://medical-transcription-app-rag.streamlit.app/

🚀 Features

Intelligent Q&A: Ask complex medical questions (e.g., "What are the symptoms of diabetes?", "How is pneumonia treated?").

Context-Aware: Uses RAG to retrieve relevant patient cases and transcriptions before generating an answer.

Source Transparency: Displays the exact medical sources (transcription chunks) used to generate the response.

Fast Inference: Powered by Groq (Llama-3.3-70b-versatile) for near-instant responses.

Efficient Search: Uses FAISS (Facebook AI Similarity Search) for high-speed vector retrieval.

🧠 How It Works (RAG Architecture)

This application follows a standard RAG pipeline:

Ingestion: The mtsamples.csv dataset (medical transcriptions) is loaded.

Chunking: Text is split into smaller chunks (500 characters) to preserve context.

Embedding: Each chunk is converted into a numerical vector using HuggingFace Embeddings (all-MiniLM-L6-v2).

Vector Store: These vectors are stored in a local FAISS index for fast searching.

Retrieval: When a user asks a question, the system finds the top 3 most similar document chunks.

Generation: The retrieved context + the user's question are sent to the Llama 3 model (via Groq), which generates an accurate answer.

🛠️ Installation & Local Setup

Follow these steps to run the app locally on your machine.

1. Clone the Repository

git clone [https://github.com/Usman3660/medical-transcription-qa-rag.git](https://github.com/Usman3660/medical-transcription-qa-rag.git)
cd medical-transcription-qa-rag


2. Create a Virtual Environment (Recommended)

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies

Important: This project uses specific versions of LangChain (v0.3.0) to ensure stability. Please use the provided requirements.txt.

pip install -r requirements.txt


4. Set Up API Keys

You will need a Groq API Key to run the LLM.

Get a free key at console.groq.com.

Enter this key in the sidebar when the app runs.

5. Run the App

streamlit run app.py


📂 Project Structure

medical-transcription-qa-rag/
├── app.py                  # Main Streamlit application code
├── requirements.txt        # Python dependencies (pinned versions)
├── mtsamples.csv           # Dataset (Medical Transcriptions)
├── faiss_index/            # Local vector store (generated on first run)
└── README.md               # Documentation


🔧 Technologies Used

Frontend: Streamlit

Framework: LangChain (v0.3.0)

LLM Provider: Groq (Llama-3.3-70b)

Embeddings: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)

Vector Database: FAISS CPU

🐛 Troubleshooting

If you encounter ModuleNotFoundError: No module named 'langchain.chains', it is likely due to a version mismatch (LangChain v1.0+ removed this module).

Solution: Ensure you install the exact dependencies listed in requirements.txt:

langchain==0.3.0
langchain-community==0.3.0
langchain-core==0.3.0


🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

📜 License

This project is licensed under the MIT License.
