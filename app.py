import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Document QA System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session states
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

def process_pdf(uploaded_file):
    """Process the uploaded PDF file and create a vector store."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create vector store using FAISS
        vector_store = FAISS.from_documents(texts, embeddings)

        # Save the vector store
        vector_store.save_local("faiss_index")

        return vector_store

    finally:
        # Clean up temporary file
        os.unlink(tmp_file_path)

def main():
    # Sidebar
    with st.sidebar:
        st.title("ℹ️ About")
        st.markdown("""
        This is a Document QA system that allows you to:
        1. Upload PDF documents
        2. Ask questions about the content
        3. Get AI-generated answers
        
        Built with:
        - Streamlit
        - LangChain
        - Hugging Face
        - FAISS
        """)

    # Main content
    st.title("📚 Document QA System")
    st.markdown("---")

    # Check for HUGGINGFACEHUB_API_TOKEN
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        st.error("⚠️ HUGGINGFACEHUB_API_TOKEN not found. Please set it in your environment variables.")
        return

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=['pdf'],
        help="Upload a PDF file to analyze"
    )

    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                st.session_state.vector_store = process_pdf(uploaded_file)
                st.success("✅ Document processed successfully!")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                return

    # Query input
    query = st.text_input(
        "Ask a question about your document:",
        placeholder="Enter your question here...",
        disabled=not st.session_state.vector_store
    )

    if query and st.session_state.vector_store is not None:
        try:
            with st.spinner("Generating answer..."):
                # Initialize HuggingFace model
                llm = HuggingFaceHub(
                    repo_id="google/flan-t5-base",
                    model_kwargs={"temperature": 0.5, "max_length": 512}
                )

                # Create QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vector_store.as_retriever()
                )

                # Get response
                response = qa_chain.run(query)
                
                # Display response
                st.markdown("### Answer:")
                st.write(response)

        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
