from utils import (
    initialize_github_client,
    create_qa_prompt_template, initialize_github_loader,
    setup_index_and_query_engine, load_environment_and_models,
    setup_index_and_chat_engine
)
from llama_index.readers.github import GithubRepositoryReader
import os


def main() -> None:
    embed_model, llm = load_environment_and_models()
    # -----------------------------------------------------
    # --------Enter Repo and File Extensions Here----------
    # -----------------------------------------------------
    owner = ""
    repo = ""
    branch = ""
    filter_file_extensions = (
        [],
        GithubRepositoryReader.FilterType.INCLUDE,
    )
    github_token = os.environ.get("GITHUB_TOKEN")
    github_client = initialize_github_client(github_token)

    loader = initialize_github_loader(github_client, owner, repo, filter_file_extensions)
    docs = loader.load_data(branch=branch)

    """Below is both a Query Engine and a Chat Engine, change response var to switch back and forth
    Chat Engine has memory adding the ability for the model to grab chat history and answer questions based off
    of old messages.
    """

    # Query Engine
    query_engine = setup_index_and_query_engine(docs, embed_model, llm)
    qa_prompt_tmpl = create_qa_prompt_template()
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

    # Chat Engine, Added Memory
    chat_engine = setup_index_and_chat_engine(docs, embed_model, llm)

    while True:
        user_query = input("Enter your question about the repository (or e to exit): ")
        if user_query.lower() == 'e':
            print("Exiting the program. Goodbye!")
            break

        response = chat_engine.chat(user_query)
        print("Response:", response)
        print("\n" + "-" * 100 + "\n")


if __name__ == "__main__":
    main()
