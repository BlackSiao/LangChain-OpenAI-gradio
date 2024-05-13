from langsmith import Client

client = Client()

# 定义数据集 粗略先来20个
dataset_name = "RAG系统测试数据集"
dataset = client.create_dataset(dataset_name, description="此数据集用来评估RAG系统，以精准度和召回率作为评估标准")
dataset_inputs = [
  "西游记的作者是谁？",
  "儒林外史里面第七回的标题是什么？",
  "骆驼祥子的工作是什么？",
  "背影的作者看的第一本书是什么？",
  "凡·高的名字叫什么？",
  "文森特为什么觉得得到调离后房间的消息很重要？"
  "祥子的老婆是谁？"
]
dataset_outputs = [
    {"answer": ["西游记的作者是吴承恩"]},
    {"answer": ["儒林外史里面第七回的标题是：范学道视学报师恩 王员外立朝敦友谊"]},
    {"answer": ["骆驼祥子的工作是洋车夫"]},
    {"answer": ["背影的作者看的第一本书是三毛流浪记"]},
    {"answer": ["凡·高的名字叫文森特"]},
    {"answer": ["因为他计划结婚了，调动后房间将对他有重大的意义。"]},
    {"ansear": ["祥子的老婆是虎子"]}
]
client.create_examples(
    inputs=[{"question": q} for q in dataset_inputs],
    outputs=dataset_outputs,
    # dataset_id=dataset.id  对此处进行一次修改
    dataset_name=dataset_name
)
