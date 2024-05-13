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
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)



# 文心一言配置
WENXIN_APP_Key = "sKzLrpmNHh4iHVGqwmntUurg"
WENXIN_APP_SECRET = "DtHAE7441OlC1g0MsWoC3eMt6UVSr1zf"
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
# 定义提示模板
template = """您是一个问答任务的助手。
使用以下检索到的上下文片段回答问题。
如果您不知道答案，只需说您不知道。
最多使用两个句子，保持答案简洁。
问题：{question}
上下文：{context}
答案：
"""
qa_prompt = ChatPromptTemplate.from_template(template)
# 加载模型
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

questions = ["Who is the wife of Camel Xiangzi?",
             "Who is the author of Journey to the West?",
             "What is the courtesy name of Dong Zhuo?",
             "What is the job of Camel Xiangzi?",
             "In the novel 'The Back', what is the first book that 'I' read?"
]
ground_truths = [["The wife of Camel Xiangzi is Huzi"],
                 ["The author of Journey to the West is Wu Cheng'en"],
                 ["The courtesy name of Dong Zhuo is Zhongying"],
                 ["The job of Camel Xiangzi is a rickshaw puller"],
                 ["In the novel 'The Back', the first book that 'I' read is 'The Story of a Stray' by Sanmao"]]


answers = []
contexts = []

for query in questions:
  answers.append(rag_chain.invoke(query))
  contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

result = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
    llm=model,
    embeddings=embedding
)

# 然后使用 df 来显示数据，会根据设置的选项显示
df = result.to_pandas()
df.to_excel('~/Desktop/result.xlsx', index=False)  # 将 DataFrame 保存为名为 result.xlsx 的 Excel 文件到桌面，不包含索引

