from langchain_wenxin import Wenxin
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from typing import List
from langsmith.schemas import Example, Run
from langsmith.evaluation import evaluate
import numpy as np

# 文心一言配置
WENXIN_APP_Key = "你的文心一言API-KEY"
WENXIN_APP_SECRET = "你的文心一言密钥"
client = Client()


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


# 设置主链的内容
qa_system_prompt = """你是一个特别擅长回答问题的助手。 \
    你熟悉经济学，历史，中国和美国文学。\
    如果你不知道答案，回答不知道即可，不要尝试捏造答案。 \
    尽可能简洁明了的回答问题。\
    """
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{question}"),
    ]
)

model = Wenxin(
    temperature=0.9,
    model="ernie-bot-turbo",
    baidu_api_key=WENXIN_APP_Key,
    baidu_secret_key=WENXIN_APP_SECRET,
    verbose=True,
)

# 将llm的输出转换为str
output_parser = StrOutputParser()
# 加载embedding模型
embedding = load_embedding_model('text2vec3')
setup_and_retrieval = RunnableParallel(
    {"question": RunnablePassthrough()}
)
rag_chain = (
 setup_and_retrieval | qa_prompt | model
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
    data="RAG系统测试数据集4",   # 数据集
    summary_evaluators=[f1_score_summary_evaluator],  # The evaluators to score the results
    experiment_prefix="文心一言测试",   # A prefix for your experiment names to easily identify them
    metadata={
      "version": "2.0.1",
    },
)
