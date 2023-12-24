import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings(api_key=API_KEY)

vector_db_dir = f"vector_db/{os.getenv('CHROMA_DB_NAME')}"

db = Chroma(embedding_function=embeddings, persist_directory=vector_db_dir)

retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)

chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever, chain_type="stuff")

while True:
    reader = input("\n>> ")
    result = chain.run(reader)
    print("\n", result)
