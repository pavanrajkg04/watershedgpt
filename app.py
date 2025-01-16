import streamlit as st
from ollama import chat
import fitz  # PyMuPDF
import pdfplumber
import os

# Define the folder where your documents are stored
DOCUMENTS_FOLDER_PATH = 'Documents'  # Update this with the path to your folder containing PDFs


# Function to extract text from PDFs using PyMuPDF (fitz)
def extract_text_from_pdfs_pymupdf(folder_path):
    text = ''
    if not os.path.exists(folder_path):
        return "Error: Folder does not exist."

    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            try:
                doc = fitz.open(pdf_path)
                for page in doc:
                    text += page.get_text()
            except Exception as e:
                st.warning(f"Failed to read PDF {filename} using PyMuPDF. Error: {e}")
                continue  # Skip the corrupted file and continue with the next one
    return text


# Function to extract text from PDFs using pdfplumber (alternative method)
def extract_text_from_pdfs_pdfplumber(folder_path):
    text = ''
    if not os.path.exists(folder_path):
        return "Error: Folder does not exist."

    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text()
            except Exception as e:
                st.warning(f"Failed to read PDF {filename} using pdfplumber. Error: {e}")
                continue  # Skip the corrupted file and continue with the next one
    return text


# Function to query the chatbot and search documents
def get_answer_from_chat_and_docs(question, folder_path):
    # Try extracting text using PyMuPDF first
    docs_text = extract_text_from_pdfs_pymupdf(folder_path)

    # If PyMuPDF fails, fall back to pdfplumber
    if docs_text == "" or "Error" in docs_text:
        st.warning("PyMuPDF extraction failed. Trying pdfplumber instead.")
        docs_text = extract_text_from_pdfs_pdfplumber(folder_path)

    if docs_text == "Error: Folder does not exist.":
        return docs_text

    # Step 2: Call the chatbot API
    response = chat(model='phi4', messages=[
        {'role': 'user',
         'content': f" you are a agriculture and watershed bot and Answer this question based on the following documents: {docs_text} \n\nQuestion: {question}"}
    ])

    # Step 3: Get the chatbot's response
    return response['message']['content']


# Streamlit UI
def main():
    st.title("Watershed-GPT 1.0")

    # Automatic fetching of documents from the folder
    folder_path = DOCUMENTS_FOLDER_PATH

    if not os.path.exists(folder_path):
        st.error(f"Documents folder '{folder_path}' does not exist!")
        return

    st.write(
        f"This is trained on small set of data. To scale this application add pdfs only with texts format to Documents folder")

    # User inputs question
    question = st.text_input("Ask me anything related to watershed karnataka:")

    if question:
        st.write(f"Searching documents for: {question}")
        with st.spinner("Finding the answer..."):
            answer = get_answer_from_chat_and_docs(question, folder_path)
            st.write(f"Answer: {answer}")


if __name__ == "__main__":
    main()
