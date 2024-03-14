from openai import OpenAI

client = OpenAI(api_key="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")

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "你是一个精通魔法的魔法师，最大的爱好就是研究新的魔法"},
    {"role": "user", "content": "为我调制一瓶能让人长高的魔药"}
  ]
)

print(completion.choices[0].message)