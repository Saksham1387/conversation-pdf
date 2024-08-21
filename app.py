import streamlit as st
from PyPDF2 import PdfReader
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import os
#sidebar contents

with st.sidebar:
    st.title('Chat with PDF')
    

def main():
    st.header("Grow Knowledge in seconds")
    load_dotenv()
    pdf=st.file_uploader("Upload your PDF",type="pdf")
    
    
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text +=page.extract_text()
        
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks=text_splitter.split_text(text=text)
        
        store_name =pdf.name[:-4]

        if os.path.exists(f"{store_name}.pk1"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore=pickle.load(f)
            
            
        else:
             embeddings=OllamaEmbeddings()
             VectorStore=FAISS.from_texts(chunks,embedding=embeddings)
             with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)
             

    query= st.text_input("ask questions about your reserach paper?")
    st.write(query)
    if query:
        docs=VectorStore.similarity_search(query=query,k=3)
        llm=Ollama(model="llama2")
        chain=load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response=chain.run(input_documents=docs,question=query)
            print(cb)
        st.write(response)


if __name__=='__main__':
    main()