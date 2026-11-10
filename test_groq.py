
from langchain_groq import ChatGroq

# Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# Send a test prompt
response = llm.invoke(
    "Explain RAG (Retrieval-Augmented Generation) in two simple sentences."
)

# Print the response
print("\nGroq Response:")
print(response.content)
