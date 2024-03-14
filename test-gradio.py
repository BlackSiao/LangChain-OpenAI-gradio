# 引入gradio为本地知识库做出一个精美的可交互web端

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import gradio as gr

# chat_function包含二个形参，分别代表本轮用户输入的问题
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
 # 提示词模板，其作用是输入问题给llm处理
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(temperature=1.0, model="gpt-3.5-turbo")
output_parser = StrOutputParser()

chain = prompt | model | output_parser
response = chain.invoke("你好")
print(response)


