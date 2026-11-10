
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize local Ollama model
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

# Create prompt
prompt = ChatPromptTemplate.from_template(
    "Answer this question briefly: {question}"
)

# Create chain
chain = prompt | llm | StrOutputParser()

# Execute chain
response = chain.invoke({
    "question": "What is an insurance claim?"
})

print("Response:", response)
