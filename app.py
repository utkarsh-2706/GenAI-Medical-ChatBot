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
import logging

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Check if API keys are set
if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing required API keys. Check your .env file.")

# Set environment variables for LangChain
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download Hugging Face embeddings
logger.info("Downloading Hugging Face embeddings...")
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone vector store
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Groq client using LangChain's ChatGroq
groq_client = ChatGroq(api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768", temperature=0.4)

# Define prompt template
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

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("msg")
        if not msg:
            return jsonify({"error": "No input provided"}), 400

        logger.info(f"User Input: {msg}")

        # Invoke the RAG chain
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "No response available")

        logger.info(f"Response: {answer}")

        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
