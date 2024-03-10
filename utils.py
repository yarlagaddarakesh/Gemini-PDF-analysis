from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

genai.configure(api_key="AIzaSyBJdSOqDwQYlpoBQ4Mt-sP33fO_R8qPDQw")
os.environ["GOOGLE_API_KEY"]="AIzaSyBJdSOqDwQYlpoBQ4Mt-sP33fO_R8qPDQw"

#Function to get response from GEMINI PRO
def get_model_response(file,query):
    raw_text = ''
    for i, page in enumerate(file.pages):
        text = page.extract_text()
        if text:
            raw_text += text


    # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text) 
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    docsearch = FAISS.from_texts(texts, embeddings)
    print(docsearch)

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