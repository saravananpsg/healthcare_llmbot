import os
import openai
# imports
import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os  # for getting API token from env variable OPENAI_API_KEY
from scipy import spatial  # for calculating vector similarities for search
# from langchain.models import OpenAIModel
from read_data import get_vector_store


data_exist=True
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]
# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


if data_exist:
    embeddings_path = 'data/doc_embedding.csv'
    df = pd.read_csv(embeddings_path)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
else:
    knowledge_base, df = get_vector_store()

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 5
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"]+"||"+row['src'], relatedness_fn(query_embedding, row["embedding"]))
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
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
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
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message, (strings, relatednesses) = query_message(query, df, model=model, token_budget=token_budget)
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
    return response_message, (strings, relatednesses)


if __name__ == "__main__":

    answer, (retreived_docs, relatednesses) = ask("what is gestational diabetes ?", df)
    print(answer)
    for docs in retreived_docs:
        doc = docs.split('||')
        source = {
            "source_doc": doc[0],
            "source": doc[1],
        }
        print(source)
