# 在这个示例中，将了解retrieval机制到底是用来干什么的，它是如何用来联系上下文的。
# 引入gradio为本地知识库做出一个精美的可交互web端
import gradio
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
import gradio as gr
import os
import time
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

"""
    加载embedding (从huggingface上下载，我采用的是本地下载)
    embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec2": "uer/sbert-base-chinese-nli",
    "text2vec3": "shibing624/text2vec-base-chinese",}
"""

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


# 定义一个函数用来作为gr.ChatInterface()的fn，history[
def predict(message, history):
    # 设置提示词，其作用是输入问题给llm处理
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    # 加载并初始化模型
    model = ChatOpenAI(temperature=1.0, model="gpt-3.5-turbo")
    # 将llm的输出转换为str
    output_parser = StrOutputParser()
    # Retriever检索函数 Vector store-backed retriever
    # docs = load_documents()
    # 加载embedding模型
    embedding = load_embedding_model('text2vec3')
    # 加载数据库，不存在向量库就生成，否则直接加载
    if not os.path.exists('VectorStore'):
        documents = load_documents()
        db = store_chroma(documents, embedding)
    else:
        db = Chroma(persist_directory='VectorStore', embedding_function=embedding)
    # 这个写法是照抄LCEL的说明，并没有完全理解
    retriever = db.as_retriever()
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    # 链式写法
    chain = setup_and_retrieval | prompt | model | output_parser
    response = chain.invoke(message)
    return response


# 本地检索，加载本地文件也应该单独领出来写一个函数

# 这里我理解为可以将web交互端的输入作为predict函数的message，并返回对应的回答
demo = gr.ChatInterface(fn=predict,
                       # examples=["给我推荐基本适合学习经济学的书", "区块链是什么？", "我喜欢一个女孩子，该如何追求她"],
                        title="本地知识库问答系统",
                        # 额外的输入，定义滑块和文件上传功能
                        )
if __name__ == "__main__":
    demo.launch()

