# LangChain-OpenAI-gradio
本地知识库问答，使用LangChain作为框架，PyCharm作为运行环境，通过简单的配置gradio完成web交互端

！！！
在运行本项目之前，请首先申请OpenAI的API_Key, 并确认能够正常的翻墙，否则无法正常运行。

# 项目目前成果
要想看成品，可以直接运行final_test.py文件，在命令栏点击链接即可跳转到Web交互端，本项目目前训练集
较少，初始训练集只包括'book'文件夹内的几本小说，可以在web端通过自己上传本地文件的情况下扩充本地知识库

# 各文件介绍
1. API-test.py ---最为简单的测试代码，用来判断是否程序能够正常的通过API链接到OpenAI的服务器
2. test-gradio ---通过gradio添加简单的交互界面，测试gradio能否正常使用
3. retrieval-test.py ---参考LangChain的官网写的本地检索器，对本地文件进行：Load-->Spliter-->Store, 之后用户的问题均从向量库VectorStore里面查找相关项来辅助llm回答
4. Contextualizing-question.py  ---尝试LangChain官网的本地检索器，添加上下文联系
5. final_test.py  ---最终成品，运行即可，实现完全体的本地知识库问答系统


成品演示：

