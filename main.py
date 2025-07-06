from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
 Hi i am pyra e a medical chatbot that provides advice on simple illnesses.

conversation history:
{context}
question: {question}
answer:
"""

model = OllamaLLM(model="llama2", temperature=0.1, max_tokens=512)
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    context = ""
    print("Medical chatbot is ready. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        result = chain.invoke({"context": context, "question": user_input})
        print("Pyra:", result)
        context += f"\nUser: {user_input}\nPyra: {result}"

if __name__ == "__main__":
    handle_conversation()
