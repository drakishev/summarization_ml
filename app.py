from langchain_ollama import ChatOllama

from models import check_if_model_is_available
from document_loader import load_documents_into_database
import argparse
import sys
from llm import getChatChain


def main(llm_model_name: str, documents_path: str) -> None:
    # Check to see if the model is available, if not attempt to pull it
    try:
        check_if_model_is_available(llm_model_name)
    except Exception as e:
        print(e)
        sys.exit()

    # Creating database from documents (now just for storage, no embeddings)
    try:
        db = load_documents_into_database(documents_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit()

    llm = ChatOllama(model=llm_model_name, model_kwargs={"enable_thinking": False})
    chat = getChatChain(llm, db)

    while True:
        try:
            user_input = input(
                "\n\nPlease enter your question (or type 'exit' to end): "
            ).strip()
            if user_input.lower() == "exit":
                break
            else:
                chat(user_input)

        except KeyboardInterrupt:
            break


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local LLM with RAG with Ollama.")
    parser.add_argument(
        "-m",
        "--model",
        default="mistral",
        help="The name of the LLM model to use.",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=".",
        help="The path to the directory containing documents to load.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.model, args.path)
