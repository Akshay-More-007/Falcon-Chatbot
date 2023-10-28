pip install langchain
pip install huggingface_hub
pip install streamlit 

import streamlit as st
from langchain import HuggingFaceHub
from apikey import apikey_hungingface
from langchain import PromptTemplate, LLMChain
import os

# Set Hugging Face Hub API token
# Make sure to store your API token in the `apikey_hungingface.py` file
os.environ["HUGGINGFACEHUB_API_TOKEN"] = apikey_hungingface

# Set up the language model using the Hugging Face Hub repository
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.3, "max_new_tokens": 2000})

# Set up the prompt template
template = """
You are now connected to the FALCON LLM ChatBot,
a highly advanced artificial intelligence assistant designed to provide comprehensive and insightful answers to your questions.
Whether you seek information, advice, or simply want to engage in a stimulating conversation, I'm here to assist you.

FALCON LLM is powered by state-of-the-art natural language processing models, and I'm equipped to handle a wide range of topics.
Feel free to ask me anything, from scientific queries to general knowledge, from technology trends to historical events.
No question is too big or too small.

For example, you can inquire about the latest advancements in AI and machine learning, request explanations of complex concepts, seek recommendations on books or movies,
or even ask for help with problem-solving or decision-making. I'm your trusted companion on this journey of knowledge and exploration.

Here's how to get the most out of our interaction:
1. Be clear and specific: The more detailed your question, the more accurate and helpful my response will be.
2. Ask follow-up questions: If you need further clarification or want to explore a topic in more depth, don't hesitate to ask for additional information.
3. Stay polite and respectful: While I'm here to assist you, it's important to maintain a courteous and respectful tone in our conversation.
Question: {question}\n\nAnswer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Create the Streamlit app
def main():
    st.title("🦜️🔗FALCON LLM ChatBot App")

    # Get user input
    question = st.text_input("Enter your question")

    # Generate the response
    if st.button("Get Answer"):
        with st.spinner("Generating Answer..."):
            response = llm_chain.run(question)
        st.success(response)

if __name__ == "__main__":
    main()
