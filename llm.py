from operator import itemgetter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string, HumanMessage, AIMessage
from langchain_core.prompts import format_document
from langchain.prompts.prompt import PromptTemplate
from langchain_core.documents import Document  # Add this import
import tiktoken
import re
 
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def remove_think_blocks(text_or_chunk):
    if hasattr(text_or_chunk, 'content'):
        text = text_or_chunk.content
    else:
        text = text_or_chunk

    if not isinstance(text, str):
        text = str(text)

    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

condense_question = """
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question)

answer = """
### Instruction:
You're a helpful research assistant, who answers questions based on provided research in a clear way and easy-to-understand way.
If there is no research, or the research is irrelevant to answering the question, simply reply that you can't answer.
Please reply with just the detailed answer and your sources. If you're unable to answer the question, do not list sources

## Research:
{context}

## Question:
{question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(answer)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="Source Document: {source}, Page {page}:\n{page_content}"
)

def _combine_documents(docs, max_tokens=3500, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    # Ensure each document has a 'page' metadata field; default to "1" if missing
    doc_strings = []
    for doc in docs:
        metadata = doc.metadata.copy()
        if "page" not in metadata:
            metadata["page"] = "1"  # Default page value
        doc_with_page = Document(page_content=doc.page_content, metadata=metadata)
        doc_strings.append(format_document(doc_with_page, document_prompt))

    combined_text = document_separator.join(doc_strings)
    token_count = count_tokens(combined_text)
    if token_count > max_tokens:
        while token_count > max_tokens and doc_strings:
            doc_strings.pop()
            combined_text = document_separator.join(doc_strings)
            token_count = count_tokens(combined_text)
    return combined_text

memory = ChatMessageHistory()

def getStreamingChain(question: str, memory_data, llm, db):
    memory.messages = [
        HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in memory_data
    ]

    retriever = db.as_retriever(search_kwargs={"k": 5})
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(
            lambda x: "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in x["memory"]]
            )
        ),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
        | (lambda x: x.content if hasattr(x, "content") else x)
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"], max_tokens=3500),
        "question": itemgetter("question"),
    }

    answer = final_inputs | ANSWER_PROMPT | llm

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    return final_chain.stream({"question": question, "memory": memory_data})

def getChatChain(llm, db):
    retriever = db.as_retriever(search_kwargs={"k": 5})

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(lambda _: get_buffer_string(memory.messages)),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
        | (lambda x: x.content if hasattr(x, "content") else x)
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"], max_tokens=3500),
        "question": itemgetter("question"),
    }

    answer = {
        "answer": final_inputs
                  | ANSWER_PROMPT
                  | llm.with_config(callbacks=[StreamingStdOutCallbackHandler()]),
    }

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    def chat(question: str):
        modified_question = f"{question} /no_think"
        inputs = {"question": modified_question}
        memory.add_message(HumanMessage(content=modified_question))
        result = final_chain.invoke(inputs)
        cleaned_response = remove_think_blocks(
            result["answer"].content if hasattr(result["answer"], "content") else result["answer"]
        )
        memory.add_message(AIMessage(content=cleaned_response))
        return cleaned_response

    return chat
