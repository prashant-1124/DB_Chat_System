from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="PowerBI.csv",
    encoding="utf-8"
)

documents = loader.load()

print("Total records:", len(documents))

# Use a text-generation model
model = OllamaLLM(model="llama3.2")

template = """
You are an expert in PowerBI, helping users understand its definitions and usage.

Here are some relevant answers: {answers}

Here is the question: {question}

Provide a clear and simple explanation.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    question = input("Ask your Question (q to quit): ")
    if question =='q':
        break
    result = chain.invoke({
    "answers": [],
    "question": "Question"
})

    print(result)