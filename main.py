from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json
import os

template = """
you are pyra. You are a kind, caring nurse. Answer as if you are gently helping your patient and you will provide advice on simple illnesses.

conversation history:{context}
question:{question}
answer:
"""

model = OllamaLLM(model="llama2:7b-q4_K_M", temperature=0.1, max_tokens=512)
# fucking maybe
#odel = OllamaLLM(model="unsloth/Phi-3-mini-4k-instruct-bnb-4bit", temperature=0.1, max_tokens=2048)
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "conversations.json")

with open(data_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

def find_relevant_example(user_question):
    
    for item in train_data:
        if user_question.lower() in item["prompt"].lower():
            return item["response"]
    return ""

def handle_conversation():
    BLUE = "\033[94m"
    RED = "\033[91m"
    RESET = "\033[0m"
    context = ""
    print("Hi am Pyra ready to answer your questions. about your health. When you want to end the conversation Type 'exit' to quit.")
    while True:
        user_input = input(f"{BLUE}User: {RESET}")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        context += f"\nUser: {user_input}\n"
        relevant = find_relevant_example(user_input)
        if relevant:
            context += f"\nRelevant example: {relevant}\n"

        result = chain.invoke({"context": context, "question": user_input})
        print(f"{RED}Pyra: {result}{RESET}")
        context += f"\nUser: {user_input}\nPyra: {result}"

if __name__ == "__main__":
    handle_conversation()

