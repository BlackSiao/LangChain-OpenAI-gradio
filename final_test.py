# 完成基本的UI界面，调试使得LLM回答的准确度提高
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
import gradio as gr
import os
import time
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage

# 调用LangChain-smith
os.environ['LANGCHAIN_API_KEY'] = 'ls__57a5dab74800476883643b1813746542'
os.environ['LANGCHAIN_PROJECT'] = 'RAG-Application'
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
chat_history = []

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
    # 加载并初始化模型
    model = ChatOpenAI(temperature=1.0, model="gpt-3.5-turbo")
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

    # 设置子链，如果用户的最新问题和前文相关，交由子链处理，在提示词内添加上chat_histor
    contextualize_q_system_prompt = """如果用户最新的提问内容和之前的对话内容
        相关，则重新编排用户的最新提问，添加上之前的对话内容；
        如果用户最新的提问内容不涉及到之前的对话内容，则直接返回该提问内容
        不用回答此问题，你的任务要么重新编排该提问，要么原样返回"""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # 设置主链的内容
    qa_system_prompt = """你是一个专门回答问题的助手。 \
        擅长使用以下的Context作为依据回答问题。 \
        如果你不知道答案，回答不知道即可，不要尝试捏造答案。 \
        尽可能详细，全面的回答问题。\
        {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    # LCEL写法，整合子链和主链
    contextualize_q_chain = contextualize_q_prompt | model | StrOutputParser()

    # 判断用户输入的问题有没有涉及上下文
    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return contextualize_q_chain
        else:
            return input["question"]

    # 对输出的context进行整理
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            RunnablePassthrough.assign(
                context=contextualized_question | retriever | format_docs
            )
            | qa_prompt
            | model
            | output_parser
    )
    question = message
    ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), ai_msg])
    return ai_msg

# UI界面
def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_text(history, text):
    history = history + [(text, None)]

    return history, gr.Textbox(value="", interactive=False)


def add_file(history, file):
    directory = os.path.dirname(file.name)  # 拿到临时文件夹
    documents = load_documents(directory)
    embedding = load_embedding_model()
    store_chroma(documents, embedding)   #
    # 将临时上传的加载好，并存到数据库里面
    history = history + [((file.name,), None)]
    return history


# 定义滑块监听事件，当用户松开滑块时，更改对llm模型的设置
def change_temperature(temperature_value):
    # 加载并初始化模型
    model = ChatOpenAI(temperature=temperature_value, model="gpt-3.5-turbo")


def change_maxtoken(token_value):
    # 加载并初始化模型
    model = ChatOpenAI(max_tokens=token_value, model="gpt-3.5-turbo")


def bot(history):
    '''
    history参数是一个记录对话历史的列表。
    每个历史记录都是一个元组，其中包含用户消息和对应的机器人回复。
    在这个列表中，最新的对话记录总是位于列表的最后一个位置，因此history[-1]表示最新的对话记录。
    '''
    message = history[-1][0]
    # 用来在终端监视程序有无正确提取到用户提出的问题
    print(message)
    # 检查文件是否上传成功，如果上传的是文件，则用户信息就是元组
    if isinstance(message, tuple):
        response = "文件上传成功！！"
    else:
        response = predict(message, history)
    # 将最新对话记录中的机器人回复部分置空
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


with gr.Blocks() as demo:
    # 定义聊天框
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
        # layout如果为"panel"显示聊天框为llm风格，"bubbles"显示为聊天气泡
        layout="bubble"
    )
    # 定义行的布局g
    with gr.Row():
        txt = gr.Textbox(
            scale=4,  # 设置与相邻元件大小的比列
            show_label=False,
            placeholder="输入您的问题，或者上传一个文件",
            container=False,
        )
        # 限定上传文件的类型只为text文件
        btn = gr.UploadButton("📁", file_types=["text"])
        # 设置三个滑块
    with gr.Column(scale=1):
        temperature_slider = gr.Slider(minimum=0, maximum=2, value=0.7, step=0.1, interactive=True, info="温度调节滑块，用来控制llm回答的随机性")
        maxtoken_slider = gr.Slider(minimum=100, maximum=1000, value=500, step=50, interactive=True, info="控制llm回答的字数")

    # 设置提交用户问题按钮的监听事件
    # 首先调用add_text()函数处理用户输入，随后传入llm模型返回回答
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    # 设置滑块的监听事件
    temperature_slider.release(change_temperature(temperature_slider.value))
    maxtoken_slider.release(change_maxtoken(maxtoken_slider.value))

    chatbot.like(print_like_dislike, None, None)


demo.queue()
if __name__ == "__main__":
    demo.launch()