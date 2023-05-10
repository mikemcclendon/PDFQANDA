import streamlit as st
from langchain.document_loaders import UnstructuredTextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os

def process_text(text, query, openai_key):
    os.environ["OPENAI_API_KEY"] = openai_key
    loader = UnstructuredTextLoader(text)
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
    return output

st.title("Text Query App")
st.write("Enter text and a query to find relevant information.")

input_text = st.text_area("Enter your text", "")

if input_text:
    query = st.text_input("Enter your query", "")
    openai_key = st.text_input("Enter your OpenAI API key", "")

    if query and openai_key:
        if st.button("Process"):
            with st.spinner("Processing..."):
                result = process_text(input_text, query, openai_key)
                st.write("Result:")
                st.write(result)
