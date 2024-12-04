from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
import os
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
import pandas,numpy,faiss,array
dashscope_llm = DashScope(
  model_name=DashScopeGenerationModels.QWEN_MAX, api_key=os.environ["DASHSCOPE_API_KEY"]
)


#read index
faiss_read_index = faiss.read_index('vectors.index')
text_to_embedding = []
df = pandas.read_csv('运动鞋店铺知识库.txt',header=None,names=['NAME'])
for index, row in df.iterrows():
    text_to_embedding.append(row['NAME'])

#chat with users
while True:
  user_input=input()
  if user_input.lower() in ["结束对话", "再见", "退出", "结束"]:
    print("感谢您的咨询，再见")
    break
  embedder = DashScopeEmbedding(
      model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
      text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
  )
  result_embeddings = embedder.get_query_embedding(user_input)
  distances, indices = faiss_read_index.search(numpy.array([result_embeddings]).astype('float32'), k=2)
  list_index=list(indices)
  print(text_to_embedding[list_index[0][0]])
