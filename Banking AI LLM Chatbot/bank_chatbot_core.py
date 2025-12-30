# import ollama
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate


# class BankChatbot:
#     def __init__(self, vectorstore):
#         self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#     def ask(self, question):
#         # Retrieve top 3 relevant chunks
#         docs = self.retriever.get_relevant_documents(question)
#         context_text = "\n".join([doc.page_content for doc in docs])

#         prompt = f"""
# You are a bank customer support assistant. Answer the question based ONLY on the following bank documents context.
# Try to provide a real human being response.
# If the answer is not in the documents, say "I am sorry, I cannot answer that based on the documents."

# Context:
# {context_text}

# Customer Question:
# {question}

# Answer:
# """
#         response = ollama.chat(
#             model="deepseek-r1:8b",
#             messages=[
#                 {"role": "system", "content": "You are a helpful bank assistant."},
#                 {"role": "user", "content": prompt},
#             ],
#         )
#         return response["message"]["content"]


import ollama


class BankChatbot:
    def __init__(self, vectorstore):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    def ask(self, question):
        docs = self.retriever.get_relevant_documents(question)

        # Limit context size
        context_text = "\n".join(doc.page_content[:800] for doc in docs)

        prompt = f"""
You are a bank customer support assistant.
Answer ONLY from the provided context.
Be concise and professional.

Context:
{context_text}

Question:
{question}

Answer:
"""

        response = ollama.chat(
            # model="deepseek-r1:8b",
            # model="mistral:7b",
            model="mistral",  # ✅ FAST MODEL
            options={
                "temperature": 0.2,
                "num_ctx": 2048,  # limit context window
                "num_predict": 200,  # limit response length
            },
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
