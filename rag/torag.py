# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "beautifulsoup4==4.13.3",
#     "jq==1.8.0",
#     "langchain==0.3.23",
#     "langchain-community==0.3.21",
#     "langchain-core==0.3.51",
#     "langchain-openai==0.3.12",
#     "langchain-text-splitters==0.3.8",
#     "langgraph==0.3.29",
#     "marimo",
#     "typing-extensions==4.13.2",
# ]
# ///

import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import getpass
    import os
    import jq
    import bs4
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings
    from langchain.chat_models import init_chat_model
    from langchain_community.document_loaders import JSONLoader
    from langchain import hub
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langgraph.graph import START, StateGraph
    from typing_extensions import List, TypedDict

    return (
        Document,
        InMemoryVectorStore,
        JSONLoader,
        List,
        OpenAIEmbeddings,
        RecursiveCharacterTextSplitter,
        START,
        StateGraph,
        TypedDict,
        WebBaseLoader,
        bs4,
        getpass,
        hub,
        init_chat_model,
        jq,
        os,
    )


@app.cell
def _(getpass, init_chat_model, os):
    # Setup Env
    if not os.environ.get("OPENAI_API_KEY"):
      os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    return (llm,)


@app.cell
def _(InMemoryVectorStore, OpenAIEmbeddings):
    # Setup Vector Store

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore(embeddings)
    return embeddings, vector_store


@app.cell
def _(JSONLoader):
    # Load the file
    loader = JSONLoader(
        file_path='./alpaca_short.json',
        jq_schema='.',
        text_content=False
    )
    docs = loader.load()
    return docs, loader


@app.cell
def _(docs):
    print(docs)
    return


@app.cell
def _(RecursiveCharacterTextSplitter, docs):
    # Split to keep context from booming for large files
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    return all_splits, text_splitter


@app.cell
def _(all_splits, vector_store):
    # Add documents to vector store
    _ = vector_store.add_documents(documents=all_splits)
    return


@app.cell
def _(Document, List, TypedDict, hub, llm, vector_store):
    # Define prompt for question-answering
    prompt = hub.pull("rlm/rag-prompt")


    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str
        thing: str

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    # Define generate
    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content, "thing": docs_content}
    return State, generate, prompt, retrieve


@app.cell
def _(START, State, StateGraph, generate, retrieve):
    # Build the app
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph, graph_builder


@app.cell
def _(graph):
    response = graph.invoke({"question": "Tell me something funny? "})
    print(response["answer"])
    print("/n")
    print(response["thing"])
    return (response,)


if __name__ == "__main__":
    app.run()
