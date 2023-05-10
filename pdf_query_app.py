import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os
import tempfile

def process_pdf_file(pdf_file, query, openai_key):
    os.environ["OPENAI_API_KEY"] = openai_key
    
       # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name
        
    loader = UnstructuredPDFLoader(pdf_file)
    pages = loader.load_and_split()

    # Filter out invalid metadata
    filtered_pages = []
    for page in pages:
        filtered_metadata = {k: v for k, v in page.metadata.items() if isinstance(v, (str, int, float))}
        page.metadata = filtered_metadata
        filtered_pages.append(page)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(filtered_pages, embeddings).as_retriever()

    docs = docsearch.get_relevant_documents(query)
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    output = chain.run(input_documents=docs, question=query)
    
     # After processing, delete the temporary file
    os.remove(tmp_file_path)
    return output


st.title("PDF Query App")
st.write("Upload a PDF file and enter a query to find relevant information in the document.")

uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_pdf:
    query = st.text_input("Enter your query", "")
    openai_key = st.text_input("Enter your OpenAI API key", "")

    if query and openai_key:
        if st.button("Process"):
            with st.spinner("Processing..."):
                result = process_pdf_file(uploaded_pdf, query, openai_key)
                st.write("Result:")
                st.write(result)
