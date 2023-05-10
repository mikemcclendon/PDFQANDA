import streamlit as st
import openai
import PyPDF2
from io import BytesIO

# Function to convert PDF file to text
def convert_pdf_to_text(file):
    pdfReader = PyPDF2.PdfFileReader(file)
    text = ""
    for page in range(pdfReader.numPages):
        text += pdfReader.getPage(page).extractText()
    return text

# Function to query OpenAI with the text and a question
def query_openai(document, question, openai_key):
    openai.api_key = openai_key

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"{document}\n\nQuestion: {question}\nAnswer:",
        temperature=0.5,
        max_tokens=150
    )

    answer = response.choices[0].text.strip()
    return answer


st.title("OpenAI PDF Query App")
st.write("Upload a PDF file and enter a query to find relevant information in the document.")

uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_pdf:
    query = st.text_input("Enter your query", "")
    openai_key = st.text_input("Enter your OpenAI API key", "")

    if query and openai_key:
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Convert the uploaded file object to a BytesIO object
                pdf_file = BytesIO(uploaded_pdf.getvalue())
                # Convert the PDF to text
                pdf_text = convert_pdf_to_text(pdf_file)
                # Query OpenAI with the text and the question
                result = query_openai(pdf_text, query, openai_key)
                st.write("Result:")
                st.write(result)
