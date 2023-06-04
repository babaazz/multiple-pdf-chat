import streamlit as sl
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from template import user_template, bot_template, css


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def chunkify(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def vecotorization(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def chat(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory    
    )
    return conversation_chain

def handle_user_input(user_input):
    response = sl.session_state.conversation({"question": user_input})
    sl.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(sl.session_state.chat_history):
        if i % 2 == 0:
            sl.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

        else:
            sl.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    
    

def main():
    load_dotenv()
    sl.set_page_config(page_title="chat with mutiple pdfs", page_icon=":books:")

    sl.write(css, unsafe_allow_html=True)

    if "conversation" not in sl.session_state:
        sl.session_state.conversation = None
    
    if "chat_history" not in sl.session_state:
        sl.session_state.chat_histroy = None

    sl.header("Chat with mutiple pdfs :books:")
    user_input = sl.text_input("Ask a question about your document")
    if(user_input):
        handle_user_input(user_input)

    with sl.sidebar:
        sl.subheader("Your Documents")
        pdf_docs = sl.file_uploader("Upload your pdfs here and click on the 'Process'", accept_multiple_files=True)

        if sl.button("Process"):
            with sl.spinner("Processing"):
                # Get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = chunkify(raw_text)
                sl.write(text_chunks)

                # Create vector store
                vectorstore = vecotorization(text_chunks)

                # Create conversation chain
                sl.session_state.conversation = chat(vectorstore)





if __name__ == '__main__':
   main()