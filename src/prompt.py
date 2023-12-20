import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings(api_key=API_KEY)

db = Chroma(
    embedding_function=embeddings, persist_directory=os.getenv("CHROMA_DB_NAME")
)


retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever, chain_type="stuff")

chain.run("Who invented a cat flap?")
