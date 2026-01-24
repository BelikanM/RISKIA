from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
INDEX_PATH = "memory/faiss_index"

if os.path.exists(INDEX_PATH):
    vectordb = FAISS.load_local(INDEX_PATH, EMBEDDINGS, allow_dangerous_deserialization=True)
else:
    vectordb = FAISS.from_texts(["Base vide"], EMBEDDINGS)
    vectordb.save_local(INDEX_PATH)

def local_search(query: str):
    docs = vectordb.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

local_knowledge_tool = Tool(
    name="Base Locale",
    func=local_search,
    description="Recherche dans la base documentaire locale FAISS"
)
