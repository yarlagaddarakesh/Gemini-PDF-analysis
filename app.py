import streamlit as st
import tempfile
from PyPDF2 import PdfReader
from utils import get_model_response
from langchain_core.messages import AIMessage,HumanMessage


# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]


def main():
    st.title("Chat with PDF file using Gemini Pro")
    uploaded_file = st.sidebar.file_uploader("You can choose a Single or Multiple PDF's",type="pdf", accept_multiple_files=True)
    if uploaded_file is not None:
        #User Input
        user_query = st.chat_input("Enter Your Question:")
        if user_query is not None and user_query != "":
            response = get_model_response(uploaded_file, user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

        #Conversation
        for message in st.session_state.chat_history:
            if isinstance(message,AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message,HumanMessage):
                with st.chat_message("Human"):
                        st.write(message.content)
                

if __name__ == "__main__":
    main()
