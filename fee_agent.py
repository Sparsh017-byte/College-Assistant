import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
NAMESPACE = "fee"
PDF_PATH = "fee_structure.pdf"

embeddings = PineconeEmbeddings(
    model="llama-text-embed-v2"
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)


def build_fee_retriever():
    stats = index.describe_index_stats()
    namespaces = stats.namespaces or {}
    namespace_stats = namespaces.get(NAMESPACE)
    vector_count = namespace_stats.vector_count if namespace_stats else 0

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=NAMESPACE
    )

    if vector_count == 0:
        print("Uploading fee structure to Pinecone...")

        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)

        vectorstore.add_documents(chunks)

        print("Fee structure uploaded successfully.")

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )