import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)

docs = TextLoader(
    f"{Path(__file__).parent.absolute()}/facts.txt"
).load_and_split(text_splitter=text_splitter)

vector_db_dir = f"vector_db/{os.getenv('CHROMA_DB_NAME')}"

Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=vector_db_dir,
)
