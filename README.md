# Building an AI-Powered Document Retrieval System with Docling and Granite

## Overview
This project demonstrates how to build an AI-powered document retrieval system using Docling for document processing, IBM Granite models for embeddings and language generation, and LangChain for workflow orchestration. It guides you through the process of handling documents, parsing them into usable formats, storing them in a vector database, and leveraging Retrieval-Augmented Generation (RAG) to enhance query responses.

### Key Technologies:
*   **Docling**: An open-source toolkit for parsing and converting documents.
*   **Granite**: IBM's state-of-the-art Large Language Models (LLMs) and Embedding models, accessible via Replicate.
*   **LangChain**: A powerful framework for building applications powered by language models.
*   **HuggingFace Embeddings**: Used for generating embedding vectors from text.
*   **ChromaDB**: A vector database for storing and retrieving embedding vectors.
*   **Replicate**: Platform for deploying and accessing AI models, used here for Granite LLMs.

## Prerequisites
*   Familiarity with Python programming.
*   Basic understanding of Large Language Models (LLMs) and Natural Language Processing (NLP) concepts.
*   A Replicate API token (to access IBM Granite models).

## Setup and Installation

To get started, first install the necessary dependencies:

```bash
! uv pip install "git+https://github.com/ibm-granite-community/utils" \
    transformers \
    langchain_classic \
    langchain_core \
    langchain_huggingface sentence_transformers \
    langchain_chroma chromadb \
    docling \
    "langchain_replicate @ git+https://github.com/ibm-granite-community/langchain-replicate.git"
Usage
1. Setting up Embedding Model
We use a Granite Embeddings model from Hugging Face to generate embedding vectors:

from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

embeddings_model_path = "ibm-granite/granite-embedding-small-english-r2"
embeddings_model = HuggingFaceEmbeddings(
    model_name=embeddings_model_path,
)
embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)
2. Setting up the LLM
We connect to a Granite LLM via Replicate. Ensure your REPLICATE_API_TOKEN is set up as a Colab secret or environment variable.

from langchain_replicate import ChatReplicate
from ibm_granite_community.notebook_utils import get_env_var

model_path = "ibm-granite/granite-4.1-8b"
model = ChatReplicate(
    model=model_path,
    replicate_api_token=get_env_var("REPLICATE_API_TOKEN"),
    model_kwargs={
        "max_tokens": 1000,
        "min_tokens": 100,
    },
)
3. Vector Database Setup
ChromaDB is used as our vector database:

from langchain_chroma import Chroma

vector_db = Chroma(embedding_function=embeddings_model)
4. Document Processing with Docling
Docling is used to download, convert, and chunk documents from specified sources (e.g., URLs). This prepares the text for embedding and storage in the vector database.

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.labels import DocItemLabel
from langchain_core.documents import Document

sources = [
    "https://www.ufc.com/news/main-card-results-highlights-winner-interviews-ufc-310-pantoja-vs-asakura",
    "https://www.abcboxing.com/wp-content/uploads/2020/02/unified-rules-mma-2019.pdf",
]

converter = DocumentConverter()

doc_id = 0
texts: list[Document] = [
    Document(page_content=chunk.text, metadata={"doc_id": (doc_id:=doc_id+1), "source": source})
    for source in sources
    for chunk in HybridChunker(tokenizer=embeddings_tokenizer).chunk(converter.convert(source=source).document)
    if any(filter(lambda c: c.label in [DocItemLabel.TEXT, DocItemLabel.PARAGRAPH], iter(chunk.meta.doc_items)))
]

print(f"{len(texts)} document chunks created")
5. Populate the Vector Database
The processed document chunks are then added to the ChromaDB vector store.

ids = vector_db.add_documents(texts[:5]) # Example with limited documents
print(f"{len(ids)} documents added to the vector database")
6. RAG with Granite
Finally, a Retrieval-Augmented Generation (RAG) pipeline is constructed using LangChain. This pipeline retrieves relevant document chunks from the vector database based on a query and feeds them to the Granite LLM as context to generate more informed answers.

from ibm_granite_community.langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Define your query and prompt template
query = "Who won in the Pantoja vs Asakura fight at UFC 310?"
prompt_template = ChatPromptTemplate.from_template(template="{input}")

# Create the RAG chain
combine_docs_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt_template,
)
retriever = vector_db.as_retriever()
rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_docs_chain,
)

# Invoke the RAG chain with your query
output = rag_chain.invoke({"input": query})
print(output['answer'])
Next Steps
Explore advanced RAG workflows for other industries.
Experiment with other document types and larger datasets.
Optimize prompt engineering for better Granite responses.
Feel free to contribute or adapt this recipe for your specific needs! ```
