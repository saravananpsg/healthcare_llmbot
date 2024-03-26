import dotenv
import openai
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import OnlinePDFLoader
# from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


config = dotenv.dotenv_values(".env")
openai.api_key = config['OPENAI_API_KEY']
openai_api_key = config['OPENAI_API_KEY']

def get_data():
    infile_path='urls_list.txt'
    with open(infile_path, 'r') as infile:
        urls_data = infile.readlines()
    data_dict = {}
    for i, url_link in enumerate(urls_data):
        url_link = url_link.strip()
        if str(url_link).endswith('pdf') or str(url_link).__contains__('ch-api'):
            loader = OnlinePDFLoader(url_link)
            text_data = loader.load()
            text_data[0].metadata['source'] = url_link
            data_dict.update({
                i: text_data
            })
        else:
            loader = WebBaseLoader(url_link)
            text_data = loader.load()
            text_data[0].metadata['source'] = url_link
            data_dict.update({
                i: text_data
            })
    return data_dict

def get_vector_store():
    data_dict = get_data()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                   chunk_overlap=0)
    all_splits_pypdf_texts = []
    all_splits_pypdf_texts_src = []
    for k, v in data_dict.items():
        text_data = data_dict[k]
        texts = text_splitter.split_documents(text_data)
        all_splits_pypdf_texts.extend([d.page_content for d in texts])
        all_splits_pypdf_texts_src.extend([d.metadata['source'] for d in texts])

    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(all_splits_pypdf_texts, embedding)

    embed_list = []
    for i, document in enumerate(all_splits_pypdf_texts):
        embedding_rec = embedding.embed_documents([document])[0]
        embed_list.append(embedding_rec)

    df = pd.DataFrame({"text": all_splits_pypdf_texts,
                       "embedding": embed_list,
                       "src": all_splits_pypdf_texts_src})
    # df.to_csv("embedding.csv")
    return vector_store, df


if __name__ == "__main__":
    vector_store, df = get_vector_store()





