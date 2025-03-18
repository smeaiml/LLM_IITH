import streamlit as st
import os
import re
import pdfplumber
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load API Key
load_dotenv()
os.getenv("GOOGLE_API_KEY")

# Configure Generative AI
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------- PDF Processing ---------------- #

def get_pdf_text_with_structure(pdf_docs):
    """Extracts structured text from PDFs by identifying chapters, subparts, and sections."""
    structured_data = []
    
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            text_buffer = ""
            current_chapter, current_subpart, current_section = None, None, None

            for page in pdf_reader.pages:
                text = page.extract_text() or ""
                
                # Detect structure
                chapter_match = re.search(r'CHAPTER\s+([A-Z0-9]+)', text, re.IGNORECASE)
                subpart_match = re.search(r'SUBPART\s+([A-Z0-9]+)', text, re.IGNORECASE)
                section_match = re.search(r'(\d+\.\S+)\s+([\w\s]+)', text)

                if chapter_match:
                    structured_data.append((current_chapter, current_subpart, current_section, text_buffer))
                    text_buffer = ""
                    current_chapter = chapter_match.group(1)

                if subpart_match:
                    structured_data.append((current_chapter, current_subpart, current_section, text_buffer))
                    text_buffer = ""
                    current_subpart = subpart_match.group(1)

                if section_match:
                    structured_data.append((current_chapter, current_subpart, current_section, text_buffer))
                    text_buffer = ""
                    current_section = section_match.group(2)

                text_buffer += text + "\n"

            if text_buffer:
                structured_data.append((current_chapter, current_subpart, current_section, text_buffer))
    
    return structured_data

# ---------------- Vector Storage ---------------- #

def get_vector_store(structured_data):
    """Stores the extracted structured text into a vector database for efficient search."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    texts = [data[3] for data in structured_data]  # Extract text content
    metadatas = [{"chapter": data[0] or "Unknown", "subpart": data[1] or "Unknown", "section": data[2] or "Unknown"} for data in structured_data]

    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local("faiss_index")

# ---------------- Question Answering ---------------- #

def get_conversational_chain():
    """Loads a question-answering chain using Google Gemini AI."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    Ensure that you include **all** points and do not skip any details.
    If the answer is not in the provided context, say: "Answer is not available in the context."

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    """Processes user input and retrieves answers based on vector search."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Retrieve most relevant documents
    docs = [(doc, score) for doc, score in new_db.similarity_search_with_score(user_question, k=5) if score > 0.5]

    if not docs:
        st.warning("No relevant answer found in the context.")
        return

    # Provide multiple choices if more than one relevant doc
    if len(docs) > 1:
        st.markdown("Multiple sections contain relevant information. Please choose one:")
        options = {
            f"Chapter {doc.metadata.get('chapter', 'Unknown')} - {doc.metadata.get('subpart', 'Unknown')} - {doc.metadata.get('section', 'Unknown')}": doc
            for doc, _ in docs
        }
        choice = st.selectbox("Select the section:", list(options.keys()))
        selected_doc = options[choice]
    else:
        selected_doc = docs[0][0]  # Select the best match

    if not selected_doc:
        st.warning("No document selected.")
        return
    
    # Debugging Info
    print("Selected Doc Attributes:", dir(selected_doc))

    # Ensure `selected_doc` has `page_content`
    page_content = getattr(selected_doc, "page_content", "")

    if not page_content:
        st.error("Error: Selected document has no content.")
        return

    # Load question-answering chain
    chain = get_conversational_chain()
    response = chain({"input_documents": [page_content], "question": user_question})

    # Display response
    st.write("### Answer:")
    st.write(response["output_text"])

