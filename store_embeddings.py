import os
from typing import List
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable. Please check your .env file.")

# Constants
MODEL_NAME = "text-embedding-ada-002"
DOC_PATH = "hr_policy.txt"
INDEX_PATH = "faiss_index"

def create_embeddings():
    """Create and store embeddings using FAISS."""
    logger.info("Reading HR policy document...")
    with open(DOC_PATH, "r", encoding='utf-8') as file:
        content = file.read()

    # Initialize tokenizer for text splitting
    tokenizer = tiktoken.get_encoding("cl100k_base")
    def tiktoken_len(text):
        tokens = tokenizer.encode(text, disallowed_special=())
        return len(tokens)

    # Split text into chunks
    logger.info("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(content)
    logger.info(f"Created {len(chunks)} text chunks")

    # Initialize embeddings
    logger.info("Creating embeddings...")
    embeddings = OpenAIEmbeddings(
        model=MODEL_NAME,
        openai_api_key=OPENAI_API_KEY
    )

    # Check if index exists
    if os.path.exists(INDEX_PATH):
        logger.info("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True  # Safe since we created the index
        )
        logger.info("FAISS index loaded successfully")
        return vectorstore

    # Create and store vectors in FAISS
    logger.info("Storing vectors in FAISS...")
    texts = [chunk for chunk in chunks]
    metadatas = [{"source": f"chunk_{i}"} for i in range(len(chunks))]
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    
    # Save the index
    vectorstore.save_local(INDEX_PATH)
    logger.info("Vector storage complete!")
    return vectorstore

if __name__ == "__main__":
    create_embeddings()
