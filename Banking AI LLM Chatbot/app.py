# import streamlit as st
# from document_loader import split_and_embed, load_documents
# from bank_chatbot_core import BankChatbot

# # Load index
# vectorstore = split_and_embed(load_documents("bank_docs"))
# bot = BankChatbot(vectorstore)

# st.title("🏦 Bank Document Chatbot")

# question = st.text_input("Ask a question about our bank products or policies:")

# if question:
#     answer = bot.ask(question)
#     st.write(answer)


import streamlit as st
from document_loader import split_and_embed, load_documents
from bank_chatbot_core import BankChatbot

st.set_page_config(page_title="Bank Chatbot", layout="centered")


@st.cache_resource
def load_vectorstore():
    docs = load_documents("bank_docs")
    return split_and_embed(docs)


vectorstore = load_vectorstore()
bot = BankChatbot(vectorstore)

st.title("🏦 Bank Document Chatbot")

question = st.text_input("Ask a question about our bank products or policies:")

if question:
    with st.spinner("Thinking..."):
        answer = bot.ask(question)
    st.write(answer)
