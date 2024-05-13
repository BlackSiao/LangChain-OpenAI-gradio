from langchain_wenxin import Wenxin
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from typing import List
from langsmith.schemas import Example, Run
from langsmith.evaluation import evaluate
import numpy as np
from langchain_community.llms import Tongyi
import dashscope



# 文心一言配置
DASHSCOPE_API_KEY= "sk-8b5a0d0d6c8a41b6ab41a5553808be85"
client = Client()


# 文件加载，直接加载本地book文件夹下的所有文件，并使用拆分器将其拆分
def load_documents(directory):
    # silent_errors可以跳过不能解码的内容
    text_loader_kwargs = {'autodetect_encoding': True}
    loader = DirectoryLoader(directory, show_progress=True, silent_errors=True, loader_kwargs=text_loader_kwargs)
    documents = loader.load()
    # 加载文档后，要使得内容变得易于llm加载，就必须把长文本切割成一个个的小文本
    # chunk_overlap使得分割后的每一个chunk都有重叠的部分，这样可以减少重要上下文的分割； add_start_index会为每一个chunk分配一个编号
    text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0, add_start_index=True)
    split_docs = text_spliter.split_documents(documents)
    # split_docs是一个包含了多个chunk的数组，print(split_docs[0])可以看到被切割后的文本
    return split_docs


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


# 把向量存储到向量库里面, 在本地产生向量数据库"VectorStore"
def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    """
    将文档向量化，存入向量数据库
    :param docs:
    :param embeddings:
    :param persist_directory:
    :return:
    """
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db


# 设置主链的内容
qa_system_prompt = """你是一个专门回答问题的助手。 \
    擅长使用以下的Context作为依据回答问题。 \
    如果用户的问题和Context无关，则忽略Context并尝试回答问题。 \
    如果你不知道答案，回答不知道即可，不要尝试捏造答案。 \
    尽可能简洁明了的回答问题。\
    {context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{question}"),
    ]
)
# 加载模型
model = Tongyi(
    temperature=0.9,
    model="qwen-turbo",
    dashscope_api_key=DASHSCOPE_API_KEY,
    verbose=True,
)

# 将llm的输出转换为str
output_parser = StrOutputParser()
# 加载embedding模型
embedding = load_embedding_model('text2vec3')
# 加载数据库，不存在向量库就生成，否则直接加载
if not os.path.exists('../VectorStore'):
    documents = load_documents(directory='book')
    db = store_chroma(documents, embedding)
else:
    db = Chroma(persist_directory='VectorStore', embedding_function=embedding)
# 从数据库中获取一个检索器，用于从向量库中检索和给定文本相匹配的内容
# search_kwargs设置检索器会返回多少个匹配的向量
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

rag_chain = (
        setup_and_retrieval
        | qa_prompt
        | model
)


def evaluate_predict(inputs: dict):
    return {"prediction": rag_chain.invoke(inputs['question'])}


# 计算余弦相似度
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def f1_score_summary_evaluator(runs: List[Run], examples: List[Example]) -> dict:
    correct_predictions = 0
    incorrect_predictions = 0

    for run, example in zip(runs, examples):
        reference = example.outputs["answer"]
        prediction = run.outputs["prediction"]

        # 提取参考答案和预测结果中的字符串
        reference_str = " ".join(reference)
        prediction_str = " ".join(prediction)

        # 计算参考答案和预测结果的语义表示
        [vec1, vec2] = embedding.embed_documents([reference_str, prediction_str])
        # 使用余弦相似度函数计算相似度
        similarity = cosine_similarity(vec1, vec2)
        print(reference_str, "相似度", similarity)
        threshold = 0.65  # 可以根据实际情况调整阈值
        # 如果相似度>=阈值，则将其视为预测成功
        if similarity >= threshold:
            correct_predictions += 1
        # 如果相似度<阈值，则视为预测失败
        else:
            incorrect_predictions += 1

    accuracy = correct_predictions / (correct_predictions + incorrect_predictions)

    return {"key": "Accuracy", "score": accuracy}


experiment_results = evaluate(
    evaluate_predict,    # RAG系统，也是我的Target
    data="RAG系统测试数据集8",   # 数据集
    summary_evaluators=[f1_score_summary_evaluator],  # The evaluators to score the results
    experiment_prefix="通译千问-retriver",   # A prefix for your experiment names to easily identify them
    metadata={
      "version": "2.0.1",
    },
)