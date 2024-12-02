# imports
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
import pandas,numpy,faiss,array
# export DASHSCOPE_API_KEY="sk-e8f3a41bf62045dcae24e84575dda67c"

# Create embeddings
# text_type=`document` to build index

text_to_embedding = []
df = pandas.read_csv('运动鞋店铺知识库.txt',header=None,names=['NAME'])
for index, row in df.iterrows():
    text_to_embedding.append(row['NAME'])

embedder = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
)
# Call text Embedding
result_embeddings = embedder.get_text_embedding_batch(text_to_embedding)
# requests and embedding result index is correspond to.
embedding_ary=numpy.empty((21,1536))
for index, embedding in enumerate(result_embeddings):
        embedding_ary[index]=embedding
index=faiss.IndexFlatL2(1536)
index.add(embedding_ary)
faiss.write_index(index, "vectors.index")
while True:
    user_input=input()
    if user_input.lower() in ["结束对话", "再见", "退出", "结束"]:
        print("感谢您的咨询，再见")
        break
    distances, indices = index.search(numpy.array([embedder.get_query_embedding(user_input)]).astype('float32'), k=2)
    # print(f"Indices of nearest neighbors: {indices}")
    # print(f"Distances: {distances}")
    list_index=list(indices)
    print(text_to_embedding[list_index[0][0]])
