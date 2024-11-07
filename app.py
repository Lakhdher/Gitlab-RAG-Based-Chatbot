from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from langchain_ollama.llms import OllamaLLM

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize embeddings and vector database
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
persist_directory = "persisted_embeddings_2"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedder)
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4})
llm = ChatGoogleGenerativeAI(
    google_api_key=google_api_key, model="gemini-1.5-pro-latest"
)
# llm = OllamaLLM(model="llama3")


# Function to get a response from the LLM
def reply(user_query):
    # Retrieve relevant documents
    relevant_docs = retriever.invoke(user_query)

    # Format the retrieved documents with content
    retrieved_context = "\n".join([f"{doc.page_content}\n" for doc in relevant_docs])

    # Create the prompt for the LLM
    llm_prompt = f"""
    You are a helpful assistant specialized in answering queries based on specific documents.
    Your role is to provide accurate, detailed answers that show how you used the context to answer.
    If you can't answer the question, you should provide a helpful response that guides the user to the right information.

    Context:
    {retrieved_context}

    Question: {user_query}
    """

    # Get the response from the language model
    response = llm.invoke(llm_prompt).content

    # Extract and format document names
    unique_document_names = {
        (doc.metadata.get("source", "Unknown")[20:], doc.metadata.get("page"))
        for doc in relevant_docs
    }

    document_references = "\n".join(
        f"- {name[:-3]}\n" for (name, page) in unique_document_names
    )

    # Combine the LLM response with document references
    final_response = f"{response}\n\n**References:**\n{document_references}"

    return final_response


st.title("📚 RAG Based Chatbot ")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Gitlab stuff!!"}
    ]

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = reply(prompt)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
