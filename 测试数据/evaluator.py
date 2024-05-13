# 尝试使用embedding模型将文本转换为向量，并计算二者的距离和相似度。
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import numpy as np


def load_embedding_model(model_name="ernie-tiny"):
    """
    加载embedding模型
    :param model_name:
    :return:
    """
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}
    return HuggingFaceEmbeddings(
        model_name=r"C:\Users\BlackSiao\Desktop\我的2024\了解最新的AI社区\OpenAI_test\text2vec",  #手动下载模型到本地'text2vec文件夹'
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


# 计算余弦相似度
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


embedding = load_embedding_model('text2vec3')
text_1=(" 等那边庄主院卧室里的灯光一熄灭，整个庄园窝棚里就泛起一阵扑扑腾腾的骚动。"  
"还在白天的时候，庄园里就风传着一件事，说是老麦哲，就是得过“中等白鬃毛”奖的那头雄猪，"
)
text_2=("在小说动物庄园里面，哪只动物获得了“中等白鬃毛”奖？")
[vec1, vec2] = embedding.embed_documents([text_1, text_2])
# 使用余弦相似度函数计算相似度
similarity = cosine_similarity(vec1, vec2)
print("余弦相似度:", similarity)