import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
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
st.title('Automated RFP Document Generator Using LangChain')

# Input fields for user-provided minimal information
title = st.text_input("Enter the RFP Title:")
start_date = st.date_input("Enter the Start Date:")
end_date = st.date_input("Enter the End Date:")
company_name = st.text_input("Enter the Company Name:")
company_address = st.text_area("Enter the Company Address:")

# Cache the model to avoid reloading
@st.cache_resource
def load_llm_model():
    return Ollama(model="llama2")

# Define a function to create and invoke the LLM chain
def generate_rfp(title, start_date, end_date, company_name, company_address):
    # Customize the Prompt Template for RFP Writing
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """
                You are a professional proposal writer specializing in RFP documents.
                Include sections such as Background Information, Project Purpose and Goals, Budget and Scope of Work, Barriers and Roadblocks, Selection Criteria, and Submission Process.
            """),
            ("user", f"Title: {title}\nStart Date: {start_date}\nEnd Date: {end_date}\nCompany Name: {company_name}\nCompany Address: {company_address}")
        ]
    )

    # Load the cached model
    llm = load_llm_model()
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser

    # Invoke the chain with user input details
    return chain.invoke({
        "title": title,
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        "company_name": company_name,
        "company_address": company_address
    })

# Generate RFP when the button is clicked
if st.button("Generate RFP Document"):
    if title and start_date and end_date and company_name and company_address:
        try:
            result = generate_rfp(title, start_date, end_date, company_name, company_address)
            st.subheader("Generated RFP Document Content:")
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please fill in all fields to generate the RFP document.")
