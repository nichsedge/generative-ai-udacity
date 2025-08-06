from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
import os

from dotenv import load_dotenv

load_dotenv(override=True)


api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")

print(base_url, api_key)

llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.0
)

# prompt = f"""most beautiful woman from Indonesia
#     """
# response = llm.invoke([HumanMessage(content=prompt)])
# print(response.content)
