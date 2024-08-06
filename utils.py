from llama_index.core.chat_engine.types import ChatMode
from llama_index.readers.github import GithubClient
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.readers.github import GithubRepositoryReader
from llama_index.core import PromptTemplate, VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv

import torch


def initialize_github_client(github_token):
    return GithubClient(github_token)


def load_embedding_model(
        model_name: str = "dunzhang/stella_en_1.5B_v5", device: str = "cuda:1"
) -> HuggingFaceBgeEmbeddings:
    model_kwargs = {"device": device}
    encode_kwargs = {
        "normalize_embeddings": True
    }  # set True to compute cosine similarity
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embedding_model


def create_qa_prompt_template():
    qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information above I want you to think step by step to answer the query in a crisp manner,\n"
        " incase case you don't know the answer say 'I don't know!'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    return PromptTemplate(qa_prompt_tmpl_str)


def setup_index_and_query_engine(docs, embed_model, llm):
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    Settings.llm = llm
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)
    return query_engine


def setup_index_and_chat_engine(docs, embed_model, llm):
    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    Settings.llm = llm
    chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.BEST,
        memory=memory,
        llm=llm,
        context_prompt=("Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above I want you to think step by step to answer \n"
                        "the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: ")
    )
    return chat_engine


def initialize_github_loader(github_client, owner, repo, filter_file_extensions):
    return GithubRepositoryReader(
        github_client,
        owner=owner,
        repo=repo,
        filter_file_extensions=filter_file_extensions,
        use_parser=False,
        verbose=False,
        concurrent_requests=20
    )


def load_environment_and_models():
    load_dotenv()
    lc_embedding_model = load_embedding_model()
    embed_model = LangchainEmbedding(lc_embedding_model)
    llm = Ollama(model="codestral:latest", request_timeout=30.0, device=set_device(1))
    return embed_model, llm


def set_device(gpu: int = None) -> str:
    if torch.cuda.is_available() and gpu is not None:
        device = f"cuda:{gpu}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device
