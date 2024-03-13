import streamlit as st
from PyPDF2 import PdfReader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore


# Function to get response from GEMINI PRO
def get_model_response(file, query):
    # get pdf text
    raw_text = get_pdf_text(file)

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    # create vector store
    docsearch = get_vectorstore(text_chunks)

    q = "Tell me about randomforest"
    records = docsearch.similarity_search(q)

    prompt_template = """
            You have to answer the question from the provided context and make sure that you provide all the details\n
            Context: {context}?\n
            Question: {question}\n

            Answer:
        """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    response = chain(
        {
            "input_documents": records,
            "question": query
        },
        return_only_outputs=True
    )
    return response['output_text']


# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]


def main():
    st.title("Chat with PDF file using Gemini Pro")
    uploaded_file = st.sidebar.file_uploader("You can choose a Single or Multiple PDF's", type="pdf", accept_multiple_files=True)

    # Display user input box only if files are uploaded
    if uploaded_file:
        user_query = st.text_input("Enter Your Question:")
        if user_query is not None and user_query != "":
            response = get_model_response(uploaded_file, user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


if __name__ == "__main__":
    main()
