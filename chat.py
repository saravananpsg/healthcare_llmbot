from main import *
from langchain_openai import OpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    )
import sys
print("Chat starting...")
print("Building embedding from the input documents")
knowledge_base, df = get_vector_store()
print("Embedding file created..")

def get_answer():
    openai_api_key = os.environ["OPENAI_API_KEY"]

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    general_system_template = r"""Use the below articles on diabetes to answer the subsequent question with respect to healthcare context. \
    If the answer is not found in the articles, write a response in an emphatic and understanding tone \
    For example: "I couldn't find an exact match for your query. Could you rephrase the questions related to healthcare ?"
     ----
    {context}
    ----
    """
    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    pdf_qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=knowledge_base.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        verbose=False,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

    yellow = "\033[0;33m"
    green = "\033[0;32m"
    white = "\033[0;39m"

    chat_history = []
    print(f"{yellow}---------------------------------------------------------------------------------")
    print('Welcome to the Health care chatBot. You are now ready to start interacting with your documents')
    print('---------------------------------------------------------------------------------')
    while True:
        query = input(f"{green}User Query: ")
        if query == "exit" or query == "quit" or query == "q" or query == "f":
            print('Exiting')
            sys.exit()
        if query == '':
            continue
        for i in range(3):
            result = pdf_qa.invoke(
                {"question": query, "chat_history": chat_history})
            chat_history.append((query, result["answer"]))

        print(f"{white}Answer: " + result["answer"])
        chat_history.append((query, result["answer"]))

get_answer()