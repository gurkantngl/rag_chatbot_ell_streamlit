import numpy as np
import streamlit as st
import pandas as pd
import time 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import ell
from dotenv import load_dotenv

load_dotenv()
# Initialize ell
#ell.init(verbose=True, store="./logdir")
ell.models.groq.register()


# Define the streaming functions
def stream_text(text:str, delay: float = 0.02):
    for word in text.split():
        yield word + " "
        time.sleep(delay)


def stream_dicts(dict_list, delay: float = 0.02):
    for dict_item in dict_list:
        yield dict_item
        time.sleep(delay)



# Define the RAG Function
@ell.simple(model="llama3-8b-8192")
def rag(query:str, personality_traits: List[str] = ["Thinking"]):
    return f"""You are an AI assistant with the following 
    personality traits: {personality_traits},
    Answer the following questions about this {query}
    
    """



# Define the RAG Function
@ell.simple(model="llama3-8b-8192")
def rag_debugger(query:str, context:str, personality_traits: List[str] = ["Thinking"]) -> str:
    return f"""You are an AI assistant with the following personality traits: {personality_traits}
            and using Retrieval-Augmented Generation (RAG) combined with Chain-of-Thought (CoT) prompting.
            RAG enhances your responses by retrieving relevant information from a knowledge base.
            You will be provided with a query and relevant context. Use this context to inform your response,
            but also draw upon your general knowledge whrn apporpriate.
            Always strive to provide accurate, helpful, and context-aware answers.
            Given the following query and relevant context, please provide a comprehensive and accurate response and
            please break down your reasoning into step-by-step intermediate steps:

            Query: {query}

            Relevant context:
            {context}

            Steps: [step1, step2, step3]

            Final Answer:
    """


def parse_cot_prompt(text: str) -> dict:
    # Initialize the response dictionary
    response = {
        "steps" : [],
        "final_answer": ""
    }

    # Split the text into steps
    steps_text = text.split("Step ")
    steps_text = steps_text[1:] 

    # Process each step
    for step_text in steps_text:
        # Extract step number and description
        step_number, description = step_text.split(": ", 1)
        step_number = int(step_number)

        # Add the step to the response dictionary
        response["steps"].append({
            "step": step_number,
            "description": description.strip()
        })

    # Find the final answer
    final_answer_index = text.find("Final Answer:")
    if final_answer_index != -1:
        final_answer = text[final_answer_index + len("Final Answer:"):].strip()
        response["final_answer"] = final_answer

    return response


# Vector Database
class VectorStore:
    def __init__(self, vectorizer, tfidf_matrix, documents):
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.documents = documents

    @classmethod
    def from_documents(cls, documents):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        return cls(vectorizer, tfidf_matrix, documents)
    

    def search(self, query: str, k: int = 2) -> List[dict]:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [
            {"document": self.documents[i], "relevance": float(similarities[i])}
            for i in top_k_indices
        ]


def load_data(file):
    df = pd.read_csv(file)
    return df


# Define Personality Traits Dictionary
personality_traits_dict = {
    "Extraverted":"Outgoing, sociable, and assertive.",
    "Introverted":"Reserved, reflective, and independent.",
    "Sensing":"Detail-oriented, practical, and factual.",
    "Intuitive":"Imaginative, innovative, and forward-thinking.",
    "Thinking":"Logical, analytical, and objective.",
    "Feeling":"Emotionally aware, empathetic, and values-driven.",
    "Judging":"Structured, planned, and decisive.",
    "Perveiving":"Flexible, spontaneous, and adaptable."
}



def about_page():
    st.subheader("About")


def ask_llm_page():
    st.subheader("Rag App")

    st.sidebar.subheader("Select Response Mode")
    response_mode = st.sidebar.selectbox("Response Mode", ["Vector Store Only", "LLM Model"])

    selected_traits = st.sidebar.multiselect("Personality Traits", list(personality_traits_dict.keys()))
    uploaded_file = st.sidebar.file_uploader("Upload Data", type=["csv", "txt"])

    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            raw_data = load_data(uploaded_file)
            if raw_data is not None:
                documents = raw_data.head()
                st.dataframe(documents)
                st.session_state.vector_store = VectorStore.from_documents(documents)
        elif uploaded_file.type == "text/plain":
            raw_data = uploaded_file.read().decode("utf-8")
            if raw_data is not None:
                documents = raw_data.split(".")
                st.session_state.vector_store = VectorStore.from_documents(documents)
                st.write(raw_data)

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))


    query = st.chat_input("Ask LLM")
    if query:
        st.session_state.messages.append({"role":"user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Thinking..."):
            if st.session_state.vector_store:
                # Burada vector_state yerine vector_store kullanılmalı
                context = st.session_state.vector_store.search(query)  # Düzeltilen satır
                if response_mode == "LLM Model":
                    # LLM Model
                    result = rag_debugger(query, context, selected_traits)
                    explanation = parse_cot_prompt(result)
                    st.session_state.messages.append({"role":"ai", "content":result})
                    with st.chat_message("ai"):
                        st.write_stream(stream_text(explanation.get("final_answer")))
                        st.write_stream(stream_dicts(context))
                        with st.expander("Debugger"):
                            st.write(explanation)

                else:
                    st.session_state.messages.append({"role":"assistant","content":context})
                    with st.chat_message("assistant"):
                        st.write(context)

            else:
                st.error("No documents loaded. Please upload a file first.")


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# Navigation
about = st.Page(about_page, title="About", icon=":material/info:")
ask_llm = st.Page(ask_llm_page, title="Ask LLM", icon=":material/chat:")

pg = st.navigation({"Home": [ask_llm], "About": [about]})

pg.run()