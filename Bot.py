#python -m pip install --upgrade streamlit-extras
# pip install --upgrade streamlit
# pip install --upgrade openai
# pip install PyMuPDF


import streamlit as st
from dotenv import load_dotenv
import pickle
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import openai
from streamlit_extras.add_vertical_space import add_vertical_space
import openai


# Load environment variables from a .env file if present
load_dotenv()

# Set OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Sidebar contents
st.set_page_config(page_title="Conversational Q&A Chatbot")

with st.sidebar:
    st.title('About :')
    st.markdown('''
    ## 
    Unlock Insights with our LLM PDF Reader Chatbot:
    ## Built by :
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models)
    ''')
    add_vertical_space(6)
    st.write('© 2024 by G Navinash. All rights reserved.')
    
## Streamlit UI
add_vertical_space(6)
st.header("Welcome to Your Personal CV Assistant!")


def main():
    add_vertical_space(2)
    st.header("Hey, Let's Chat ...")

    store_path = os.path.abspath(r"pkl_file")
    vector_stores = []

    if os.path.exists(store_path):
        for filename in os.listdir(store_path):
            if filename.endswith(".pkl"):
                file_path = os.path.join(store_path, filename)
                with open(file_path, "rb") as f:
                    vector_store = pickle.load(f)
                    vector_stores.append(vector_store)
    else:
        st.warning("Vector data not found. Please upload PDFs first.")

    query = st.text_input("Ask Your CV Related Questions  : ")

    if query and vector_stores:
        for selected_vector_store in vector_stores:
            # Search for relevant documents using semantic similarity
            docs = selected_vector_store.similarity_search(query=query, k=3)

            # Use OpenAI for question-answering
            llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

        st.write(response)

    # PDF extraction and embedding
    directory_path = r'input_file'
    directory_contents = os.listdir(directory_path)
    text = ""

    for item in directory_contents:
        pdf_path = os.path.join(directory_path, item)

        if item.endswith('.pdf'):
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                pdf_file.close()  # Close the PDF file after use

            save_directory = r'input_file_text'

            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            save_txt_path = os.path.join(save_directory, f"{item.replace('.pdf', '')}.txt")
            with open(save_txt_path, 'w', encoding='utf-8') as file:
                file.write(text)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )

            chunks = text_splitter.split_text(text=text)

            # EMBEDDING
            store_name, _ = os.path.splitext(item)
            store_path = os.path.join(r'pkl_file', f"{store_name}.pkl")

            if os.path.exists(store_path):
                with open(store_path, "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key)
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(store_path, "wb") as f:
                    pickle.dump(VectorStore, f)

if __name__ == '__main__':
    main()











