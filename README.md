# LangChain-OpenAI-gradio
本地知识库问答，使用LangChain作为框架，PyCharm作为运行环境，通过简单的配置gradio完成web交互端

！！！
1.在运行本项目之前，请首先申请OpenAI的API_Key, 并确认能够正常的翻墙，否则无法正常运行。
2.如果想要使用LangSmith进行项目的评估，也需要注册LangChain的API_KEY,并将其添加到环境变量中，具体参考:https://docs.smith.langchain.com/tracing/quick_start

# 项目目前成果
要想看成品，可以直接运行final_test.py文件，在命令栏点击链接即可跳转到Web交互端，本项目目前训练集
较少，初始训练集只包括'book'文件夹内的几本小说，可以在web端通过自己上传本地文件的情况下扩充本地知识库

# 各文件夹介绍
** 搭建步骤文件夹
1. API-test.py ---最为简单的测试代码，用来判断是否程序能够正常的通过API链接到OpenAI的服务器
2. test-gradio ---通过gradio添加简单的交互界面，测试gradio能否正常使用
3. retrieval-test.py ---参考LangChain的官网写的本地检索器，对本地文件进行：Load-->Spliter-->Store, 之后用户的问题均从向量库VectorStore里面查找相关项来辅助llm回答
4. Contextualizing-question.py  ---尝试LangChain官网的本地检索器，添加上下文联系
** 测试数据文件夹
1.RAGAs测试.py ----使用正儿八经的测试框架测试RAG应用，提供四种评判指标，运行后在桌面生成测试结果
2.其余文件均使用链接LangSmith的方式进行评测，需要实现导入数据集
**最终成品
LLM-based-Knowledge-Base-QA.py---经过一系列测试完成的最终版，提供各种参数供用户设置，可回答本地文件夹book内的问题

# 最终成品演示：
![Image text](https://github.com/BlackSiao/LangChain-OpenAI-gradio/blob/main/img/最终demo.png)

使用LangChain-smith进行调试，可以清晰的看到本地检索了哪些段落，整个chain的工作流程

![Image text](https://github.com/BlackSiao/LangChain-OpenAI-gradio/blob/main/img/smith.png)




