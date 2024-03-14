# API-test.py的作用是了解最简单的LCEL写法，通过调用Openai的API完成，选用模型为："gpt-3.5-turbo"

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 提示词模板，其作用是输入问题给llm处理
prompt = ChatPromptTemplate.from_template("你是一个学识渊博的大魔法，接下来请传授我{magic}的奥妙")
model = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

print(chain.invoke({"magic": "火球术"}))