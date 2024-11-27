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
    # if (
    #     embedding is None
    # ):  # if the the correspondence request is embedding failed.
    #     print("The %s embedding failed." % text_to_embedding[index])
    # else:
    #     print("%d Dimension of embeddings: %s" % (index,len(embedding)))
    #     print(
    #         "Input: %s, embedding is: %s"
    #         % (text_to_embedding[index], embedding[:5])
    #      )embedding_ary[index]=embedding
index=faiss.IndexFlatL2(1536)
index.add(embedding_ary)
faiss.write_index(index, "ectors.index")
