from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
import os

dashscope_llm = DashScope(
  model_name=DashScopeGenerationModels.QWEN_MAX, api_key=os.environ["DASHSCOPE_API_KEY"]
)
while True:
  user_input=input()
  if user_input.lower() in ["结束对话", "再见", "退出", "结束"]:
    print("感谢您的咨询，再见")
    break
  responses = dashscope_llm.stream_complete(user_input.lower())
  for response in responses:
    print(response.delta, end="")
  print()