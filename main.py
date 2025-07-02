from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

response = model.invoke(messages)

print(response.content)

system_template = "Traduza do português para o {idioma}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{texto}")]
)

prompt = prompt_template.invoke({"idioma": "japonês", "texto": "Olá, tudo bem?"})
response = model.invoke(prompt)
print(response.content)