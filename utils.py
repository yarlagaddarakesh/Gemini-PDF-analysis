from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key="GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"]="GEMINI_API_KEY"


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
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore


#Function to get response from GEMINI PRO
def get_model_response(file,query):
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

    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
        
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.9)
        
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
        
    response = chain(
        {
            "input_documents":records,
            "question":query
        },
        return_only_outputs=True
    )
    return response['output_text']