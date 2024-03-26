"""
LLM based QA Retriever bot
"""
# import site-packages and modules
import os
import sys
import uuid
import json
import dotenv
import openai
import argparse
import ast  # for converting embeddings saved as strings back to arrays
from langchain_community.chat_models.openai import ChatOpenAI
from openai import OpenAI  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from read_data import get_vector_store
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils import write_json

# set config/parameters
config = dotenv.dotenv_values(".env")
openai.api_key = config['OPENAI_API_KEY']

# set model configs
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# search related docs using cosine similarity between query and all embeddings
# retrieve top-n related docs(set to 3 here - as the samples are small &
# to avoid hallucination)
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 3
) -> tuple[list[str], list[float]]:
    """
    Returns a list of strings and relatedness,
    sorted from most related to least.
    """
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"]+"||"+row['src'], relatedness_fn(query_embedding,
                                                     row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
) -> tuple:
    """
    Return a message for GPT, with relevant source texts pulled from a dataframe.
    """
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    # print(strings,relatednesses)
    introduction = """
    Use the below articles on diabetes to answer the subsequent question with respect \
    to healthcare context. If the answer cannot be found in the articles, write a \
    response in an emphatic and understanding tone For example: "I couldn't find \
    an exact match for your query. Could you rephrase the questions related to diabetes ?" 
    """

    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nNext article:\n"""\n{string}\n"""'
        if (
                num_tokens(message + next_article + question, model=model)
                > token_budget
        ):
            break
        else:
            message += next_article
    return message + question, (strings, relatednesses)


def ask(
        query: str,
        df: pd.DataFrame,
        model: str = GPT_MODEL,
        token_budget: int = 4096 - 500,
        print_message: bool = False,
) -> str:
    """
    Answers a query using GPT and a dataframe
    of relevant texts and embeddings."""
    message, (strings, relatednesses) = query_message(query, df,
                                                      model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the diabetes."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content

    source_list = []
    for i, docs in enumerate(strings):
        doc = docs.split('||')
        source_list.append({
            "source_doc": doc[0],
            "source": doc[1],
            "relatednesses_score": relatednesses[i]
        })
    resp_dict = {
        "user_query": query,
        "answer": response_message,
        "source_docs": source_list
    }

    api_filename = "gpt_3.5_"+str(uuid.uuid4())
    api_resp_file = os.path.join("results", api_filename)
    write_json(api_resp_file, resp_dict)

    return json.dumps(resp_dict, indent=4)


def get_answer():

    knowledge_base, df = get_vector_store()

    # llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    pdf_qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=knowledge_base.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        verbose=False
    )
    yellow = "\033[0;33m"
    green = "\033[0;32m"
    white = "\033[0;39m"

    chat_history = []
    print(f"{yellow}---------------------------------------------------------------------------------")
    print('Welcome to the Healthcare Bot. You are now ready to start interacting..')
    print('---------------------------------------------------------------------------------')
    while True:
        query = input(f"{green}Prompt: ")
        if query == "exit" or query == "quit" or query == "q" or query == "f":
            print('Exiting')
            sys.exit()
        if query == '':
            continue
        result = pdf_qa.invoke(
            {"question": query, "chat_history": chat_history})
        print(f"{white}Answer: " + result["answer"])
        result_dict = {
            "question": query,
            "answer": result["answer"]
        }
        api_filename = "qachain_gpt_3.5_" + str(uuid.uuid4())
        api_resp_file = os.path.join("results", api_filename)
        write_json(api_resp_file, result_dict)
        chat_history.append((query, result["answer"]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='QA retreiver chatbot')
    parser.add_argument('--query', metavar='query', required=True,
                        help='user query')
    parser.add_argument('--db_persist', metavar='db_persist',default=True,
                        help='set to true to load the embedding file ')
    args = parser.parse_args()
    query = args.query
    db_persist = args.db_persist

    # load/persist embedding file before making request
    # set db_persist if true load embedding file,
    # else create new embedding file
    if db_persist:
        embeddings_path = 'data/doc_embedding.csv'
        df = pd.read_csv(embeddings_path)
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
    else:
        knowledge_base, df = get_vector_store()
    # from gpt-3.5 with prompting and context - top-k related docs
    response = ask(query, df)
    print(response)




