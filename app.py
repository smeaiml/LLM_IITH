import streamlit as st
import os
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract and Structure Text with Section Headers
def extract_structured_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Split by subpart headers like G, G1, G2, etc.
def split_by_sections(text):
    pattern = r"(Subpart\s+[A-Z]+(?:\d*)?)"  # Matches Subpart G, G1, G2, etc.
    parts = re.split(pattern, text)
    structured_chunks = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        content = parts[i+1].strip()
        structured_chunks.append(f"{header}\n{content}")
    return structured_chunks

# Create Vector Store
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Gemini Chain
def get_conversational_chain():
    prompt_template = """
Use the context below to answer the user's question accurately. Be sure to refer to any nested subpart (e.g., G1, G2) clearly. Do not guess.

If the answer is not found in the context, respond with:
"Answer is not available in the context."

Context:
{context}

Question: {question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Answer Retrieval
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.markdown("### Reply:")
    st.markdown(f"{response['output_text']}")

# Streamlit App
def main():
    st.set_page_config(page_title="IntelliPDF", layout="centered")
    st.title("📄 IntelliPDF – Ask Questions from Complex PDFs")

    user_question = st.text_input("🔍 Ask a question:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.header("📂 Upload & Process")
        pdf_docs = st.file_uploader("Upload PDF(s)", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("🔄 Processing..."):
                raw_text = extract_structured_text(pdf_docs)
                structured_chunks = split_by_sections(raw_text)
                get_vector_store(structured_chunks)
                st.success("✅ Done processing!")

if __name__ == "__main__":
    main()
