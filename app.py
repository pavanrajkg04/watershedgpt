import streamlit as st
from ollama import chat
import fitz  # PyMuPDF
import os

# Define the folder where your documents are stored
DOCUMENTS_FOLDER_PATH = 'documents'  # Update this with the path to your folder containing PDFs


# Function to extract text from all PDFs in a folder
def extract_text_from_pdfs(folder_path):
    text = ''
    # Check if the folder exists
    if not os.path.exists(folder_path):
        return "Error: Folder does not exist."

    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
    return text


# Function to query the chatbot and search documents
def get_answer_from_chat_and_docs(question, folder_path):
    # Step 1: Search PDF files for relevant content
    docs_text = extract_text_from_pdfs(folder_path)

    if docs_text == "Error: Folder does not exist.":
        return docs_text

    # Step 2: Call the chatbot API
    response = chat(model='phi4', messages=[
        {'role': 'user',
         'content': f"Answer this question based on the following documents: {docs_text} \n\nQuestion: {question}"}
    ])

    # Step 3: Get the chatbot's response
    return response['message']['content']


# Streamlit UI
def main():
    st.title("Chat with Your PDF Documents")

    # Automatic fetching of documents from the folder
    folder_path = DOCUMENTS_FOLDER_PATH

    if not os.path.exists(folder_path):
        st.error(f"Documents folder '{folder_path}' does not exist!")
        return

    st.write(f"Documents are being fetched from: {folder_path}")

    # User inputs question
    question = st.text_input("Ask a question about your documents:")

    if question:
        st.write(f"Searching documents for: {question}")
        with st.spinner("Finding the answer..."):
            answer = get_answer_from_chat_and_docs(question, folder_path)
            st.write(f"Answer: {answer}")


if __name__ == "__main__":
    main()
