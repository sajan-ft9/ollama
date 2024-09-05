import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()
api_key = os.getenv("LANGCHAIN_API_KEY")

# Check if the environment variable is set
if not api_key:
    st.error("LANGCHAIN_API_KEY is not set in the environment variables.")
    st.stop()

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = api_key

# Streamlit framework
st.title('AI Assistant Using LangChain')

# Input field for user question
question = st.text_input("Enter your question:")

# Cache the model to avoid reloading
@st.cache_resource
def load_llm_model():
    return Ollama(model="llama2")

# Define a function to create and invoke the LLM chain
def generate_response(question):
    # Customize the Prompt Template for Q&A
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please respond concisely to the user queries."),
            ("user", f"Question: {question}")
        ]
    )

    # Load the cached model
    llm = load_llm_model()
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser

    # Invoke the chain with user input
    response = chain.invoke({"question": question})

    # Return a concise response
    return response.strip()  # Stripping any extra spaces or newlines

# Generate response when the button is clicked
if st.button("Get Answer"):
    if question:
        try:
            with st.spinner("Generating response..."):
                result = generate_response(question)
            st.subheader("Assistant's Response:")
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a question to get a response.")
