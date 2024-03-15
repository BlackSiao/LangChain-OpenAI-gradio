# 在这个示例中，将了解retrieval机制到底是用来干什么的，它是如何用来联系上下文的。
# 引入gradio为本地知识库做出一个精美的可交互web端
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


# 文件加载，直接加载本地book文件夹下的所有文件，并使用拆分器将其拆分
def load_documents(directory='book'):
    # silent_errors可以跳过不能解码的内容
    text_loader_kwargs = {'autodetect_encoding': True}
    loader = DirectoryLoader('book', show_progress=True, silent_errors=True, loader_kwargs=text_loader_kwargs)
    documents = loader.load()

    # 加载文档后，要使得内容变得更加易于llm加载，就必须把长文本切割成一个个的小文本
    # Split by tokens
    text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    split_docs = text_spliter.split_documents(documents)
    # print(split_docs[0])可以看到被切割后的文本
    return split_docs


# 使用OpenAI的embedding模型要钱，之后可以换，现在还是用本地的吧
# embedding的作用是把文本转换到向量空间，这样就可以进行相似化检索等内容
def load_embedding_model(model_name="ernie-tiny"):
    """
    加载embedding模型
    :param model_name:
    :return:
    """
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"} # 用GPU
    local_model_path = "D:\Hugface"  # 修改模型为本地地址
    return HuggingFaceEmbeddings(
        model_name=local_model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


# 把向量存储到向量库里面, docs是被spliter之后的列表，embedding是选择模型，persist_directory在本地产生向量数据库
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


# 定义一个函数用来作为gr.interfact()的fn，
def predict(message,history):
    # 加载并初始化模型
    model = ChatOpenAI(temperature=1.0, model="gpt-3.5-turbo")

    # 设置提示词，其作用是输入问题给llm处理
    template = """你是一名博学多才的图书馆管理员，对世界范围内的文学著作都如数家珍，
    你可以基于接下来的{content}和Question：{Question}来准确的告诉读者，这本书的简介。
    对于你不了解的，你也会诚实的告诉问答者你不知道，而不是胡编乱造
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 将llm的输出转换为str
    output_parser = StrOutputParser()

    # Retriever检索函数 Vector store-backed retriever
    docs = load_documents()
    embedding = load_embedding_model()
    db = store_chroma(docs, embedding)
    retriever = db.as_retriever()
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    # 链式写法
    chain = setup_and_retrieval | prompt | model | output_parser

    response = chain.invoke({"Question": message})
    return response


# 这里我理解为可以将web交互端的输入作为predict函数的message，并返回对应的回答
demo = gr.ChatInterface(fn=predict,
                        examples=["今天天气如何？", "区块链是什么？", "我喜欢一个女孩子，该如何追求她"],
                        title="本地知识库问答系统")
if __name__ == "__main__":
    demo.launch()
