import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
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

# Initialize memory to store conversation history
@st.cache_resource
def initialize_memory():
    return ConversationBufferMemory(return_messages=True)

memory = initialize_memory()

# Cache the model to avoid reloading
@st.cache_resource
def load_llm_model():
    return Ollama(model="llama2")

# Define a function to get session history
def get_session_history():
    return memory.chat_memory  # Return the stored chat messages

# Define a function to create and invoke the LLM chain with chat history
def generate_response(question):
    # Customize the Prompt Template for Q&A with history
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You're an assistant who speaks in English. Respond concisely."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ]
    )

    # Load the cached model
    llm = load_llm_model()
    output_parser = StrOutputParser()
    
    # Create a runnable chain with history
    runnable = prompt_template | llm | output_parser
    runnable_with_history = RunnableWithMessageHistory(
        runnable=runnable,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    # Invoke the chain with user input and update memory
    response = runnable_with_history.invoke({"input": question})
    memory.chat_memory.add_message({"role": "user", "content": question})
    memory.chat_memory.add_message({"role": "assistant", "content": response})

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