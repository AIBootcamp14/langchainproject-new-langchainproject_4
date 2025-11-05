from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings  # 또는 사용한 embedding 모델

# ✅ 동일한 embedding 모델을 다시 지정해야 함
#embedding = OpenAIEmbeddings()  # 또는 이전에 사용한 embedder
#embedder = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

# ✅ 저장된 DB 로드
#db = FAISS.load_local("faiss_langchain_db2", embedding)
db = FAISS.load_local("faiss_langchain_db2", embedding=embedder)



