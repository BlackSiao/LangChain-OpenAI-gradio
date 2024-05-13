# 配置好OpenAI-key后用于测试是否能正常连接
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import gradio as gr


# chat_function包含二个形参，分别代表本轮用户输入的问题
template = """你是一名博学多才的图书馆管理员，对世界范围内的文学著作都如数家珍，
你可以根据{Question}来准确的告诉读者，这本书的简介。对于你不了解的，你也会
诚实的告诉问答者你不知道，而不是胡编乱造
"""
 # 提示词模板，其作用是输入问题给llm处理
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(temperature=1.0, model="gpt-3.5-turbo")
output_parser = StrOutputParser()

chain = prompt | model | output_parser
print(chain.invoke({"Question": "堂吉诃德"}))
