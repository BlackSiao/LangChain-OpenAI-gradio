import openai
from langsmith.wrappers import wrap_openai
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from typing import List
from langsmith.schemas import Example, Run
from langsmith.evaluation import evaluate
from Levenshtein import distance

client = Client()

# 定义数据集 粗略先来20个
dataset_name = "RAG系统测试数据集"
dataset = client.create_dataset(dataset_name, description="此数据集用来评估RAG系统，以精准度和召回率作为评估标准")
dataset_inputs = [
  "西游记的作者是谁？",
  "儒林外史里面第七回的标题是什么？",
  "骆驼祥子的工作是什么？",
  "背影的作者看的第一本书是什么？",
]
dataset_outputs = [
    {"answer": ["吴承恩"]},
    {"answer": ["范学道视学报师恩 王员外立朝敦友谊"]},
    {"answer": ["洋车夫"]},
    {"answer": ["三毛流浪记"]},
]
client.create_examples(
    inputs=[{"question": q} for q in dataset_inputs],
    outputs=dataset_outputs,
    # dataset_id=dataset.id  对此处进行一次修改
    dataset_name=dataset_name
)

# 修改Target
openai_client = wrap_openai(openai.Client())
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


# 加载并初始化模型
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=1.0)
# 将llm的输出转换为str
output_parser = StrOutputParser()
# 加载embedding模型
embedding = load_embedding_model('text2vec3')
# 加载数据库，不存在向量库就生成，否则直接加载
if not os.path.exists('VectorStore'):
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

# 设置主链的内容
qa_system_prompt = """你是一个专门回答问题的助手。 \
    擅长使用以下的Context作为依据回答问题。 \
    如果用户的问题和Context无关，则忽略Context并尝试回答问题。 \
    如果你不知道答案，回答不知道即可，不要尝试捏造答案。 \
    尽可能详细，全面的回答问题。\
    {context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{question}"),
    ]
)
rag_chain = (
        setup_and_retrieval
        | qa_prompt
        | model
)


def evaluate_predict(inputs: dict):
    return {"prediction": rag_chain.invoke(inputs['question']).content}


# 定义评估器
def f1_score_summary_evaluator(runs: List[Run], examples: List[Example]) -> dict:
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for run, example in zip(runs, examples):
        reference = example.outputs["answer"]
        prediction = run.outputs["prediction"]

        # 计算预测结果与参考答案之间的编辑距离
        edit_distance = distance(prediction, reference)

        # 根据编辑距离设置阈值，当编辑距离小于等于阈值时，将其视为匹配
        threshold = 3  # 可以根据实际情况调整阈值

        # 如果编辑距离小于等于阈值，将其视为匹配，计为真正例
        if edit_distance <= threshold:
            true_positives += 1
        # 否则，根据参考答案和预测结果的是否存在来判断是否为假正例或假负例
        elif reference and not prediction:
            false_negatives += 1
        elif prediction and not reference:
            false_positives += 1

    if true_positives == 0:
        return {"key": "f1_score", "score": 0.0}

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return {"key": "f1_score", "score": f1_score}

experiment_results = evaluate(
    evaluate_predict,    # RAG系统，也是我的Target
    data=dataset_name,   # 数据集
    summary_evaluators=[f1_score_summary_evaluator],  # The evaluators to score the results
    experiment_prefix="rap-generator",   # A prefix for your experiment names to easily identify them
    metadata={
      "version": "2.0.0",
    },
)
