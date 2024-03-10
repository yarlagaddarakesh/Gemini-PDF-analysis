import streamlit as st
import tempfile
from PyPDF2 import PdfReader
from utils import get_model_response

def main():
    st.title("Chat with PDF file using Gemini Pro")
    uploaded_file = st.sidebar.file_uploader("Chhose a CSV file",type="pdf")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

            reader = PdfReader(tmp_file_path)

            user_input = st.text_input("Enter Your Question:")

            if user_input:
                response = get_model_response(reader, user_input)
                st.write(response)

if __name__ == "__main__":
    main()
