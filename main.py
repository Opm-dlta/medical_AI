from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json
import os
import gradio as gr

template = """
You are pyra. You are a kind, caring nurse. Answer as if you are gently helping your patient and you will provide advice on simple illnesses(your only allow to answer question related to medical if the question is not related to medical problems answer it with "I'm sorry, I can't assist with that.""do not discrib your action onlu talk").

conversation history:{context}
question:{question}
answer:
"""
model = OllamaLLM(model="llama2:7b-chat-q4_K_M", temperature=0.1, max_tokens=1048)
#model = OllamaLLM(model="llama3.3", temperature=0.1, max_tokens=1048)
#model = OllamaLLM(model="mistral", temperature=0.1, max_tokens=1048)
#model = OllamaLLM(model="unsloth/Phi-3-mini-4k-instruct-bnb-4bit", temperature=0.1, max_tokens=1048)
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

# Adapt handler to Gradio 'messages' format
chat_history = []

def to_message_tuple(turn):
    # convert (user, bot) tuple to Gradio messages format
    user_msg = {"role": "user", "content": turn[0]}
    bot_msg = {"role": "assistant", "content": turn[1]}
    return user_msg, bot_msg

async def gradio_chat(user_input, history):
    # history is a list of message dicts when using type='messages'
    context = ""
    # reconstruct simpler tuple history for find_relevant_example
    tuple_history = []
    for msg in (history or []):
        if msg["role"] == "user":
            tuple_history.append((msg["content"], ""))
        elif msg["role"] == "assistant" and tuple_history:
            # attach assistant content to last user
            last = tuple_history.pop()
            tuple_history.append((last[0], msg["content"]))

    for turn in tuple_history:
        context += f"\nUser: {turn[0]}\nPyra: {turn[1]}"

    relevant = find_relevant_example(user_input)
    if relevant:
        context += f"\nRelevant example: {relevant}\n"

    # call the model (synchronous invoke wrapped in async)
    result = chain.invoke({"context": context, "question": user_input})

    # append both messages to history and return the updated messages list
    user_msg = {"role": "user", "content": user_input}
    bot_msg = {"role": "assistant", "content": result}
    new_history = (history or []) + [user_msg, bot_msg]
    return new_history


with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", neutral_hue="gray")) as demo:
    gr.Markdown("""
    # Pyra Medical Chatbot
    <div style='background-color: white; padding: 10px; border-radius: 10px; color: #000000;'>
    <b style='color: #000000;'>Ask Pyra about simple illnesses. Pyra will only answer medical questions.</b>
    </div>
    """)

    # Use the new messages format
    chatbot = gr.Chatbot(
        label="Pyra Chat",
        type='messages',
        show_copy_button=True,
        height=400,
        elem_id="pyra-chatbot",
    )

    user_input = gr.Textbox(label="Your question", placeholder="Type your medical question here...", lines=2)
    send_btn = gr.Button("Send")
    clear_btn = gr.Button("Clear Chat")

    def clear_chat():
        return []

    # wire up buttons
    send_btn.click(gradio_chat, [user_input, chatbot], chatbot)
    clear_btn.click(lambda: [], None, chatbot)


def main():
    # Launch Gradio locally only when script executed directly
    demo.launch(share=False)


if __name__ == "__main__":
    main()

handle_conversation()

