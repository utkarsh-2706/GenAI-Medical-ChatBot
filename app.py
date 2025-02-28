from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq  # Use LangChain's Groq integration
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

# Load environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Set environment variables for LangChain
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Download Hugging Face embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index name
index_name = "medicalbot"

# Initialize Pinecone vector store
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create a retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Groq client using LangChain's ChatGroq
groq_client = ChatGroq(api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768", temperature=0.4)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the document chain
question_answer_chain = create_stuff_documents_chain(groq_client, prompt)

# Create the retrieval-augmented generation (RAG) chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    # Get user input from the form
    msg = request.form["msg"]
    print("User Input:", msg)

    # Invoke the RAG chain
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])

    # Return the response to the frontend
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)