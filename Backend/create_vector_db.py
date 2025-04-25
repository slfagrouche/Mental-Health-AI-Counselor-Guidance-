import pandas as pd
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
import os
import getpass
import shutil
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import kagglehub
from pathlib import Path
from langsmith import Client, traceable

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize NLTK and VADER
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

# Set OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI API Key: ")

# Set LangSmith environment variables
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter LangSmith API Key: ")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "MentalHealthCounselorPOC"

# Initialize LangSmith client
langsmith_client = Client()

# Define default paths
DEFAULT_OUTPUT_DIR = os.environ.get("MH_OUTPUT_DIR", "mental_health_model_artifacts")
DEFAULT_DATASET_PATH = os.environ.get("MH_DATASET_PATH", None)

# Parse command-line arguments (ignore unknown args for Jupyter/Colab)
import argparse
parser = argparse.ArgumentParser(description="Create ChromaDB vector database for Mental Health Counselor POC")
parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help="Directory for model artifacts")
parser.add_argument('--dataset-path', default=DEFAULT_DATASET_PATH, help="Path to train.csv (if already downloaded)")
args, unknown = parser.parse_known_args()  # Ignore unknown args like -f

# Set paths
output_dir = args.output_dir
chroma_db_path = os.path.join(output_dir, "chroma_db")
dataset_path = args.dataset_path

# Text preprocessing function
@traceable(run_type="tool", name="Clean Text")
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z']", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in STOPWORDS and len(tok) > 2]
    return " ".join(tokens)

# Response categorization function
@traceable(run_type="tool", name="Categorize Response")
def categorize_response(text):
    text = str(text).lower()
    labels = []
    if re.search(r"\?$", text.strip()):
        return "Question"
    if any(phrase in text for phrase in ["i understand", "that sounds", "i hear"]):
        labels.append("Validation")
    if any(phrase in text for phrase in ["should", "could", "try", "recommend"]):
        labels.append("Advice")
    if not labels:
        sentiment = analyzer.polarity_scores(text)
        if sentiment['compound'] > 0.3:
            labels.append("Empathetic Listening")
        else:
            labels.append("Advice")
    return "|".join(labels)

# Load dataset
@traceable(run_type="tool", name="Load Dataset")
def load_dataset():
    try:
        if dataset_path and os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
        else:
            # Download dataset using kagglehub
            dataset = kagglehub.dataset_download("thedevastator/nlp-mental-health-conversations", path="train.csv")
            df = pd.read_csv(dataset)
        print("First 5 records:\n", df.head())
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

# Main vector database creation
@traceable(run_type="chain", name="Create Vector Database")
def create_vector_db():
    df = load_dataset()

    # Validate and clean dataset
    if not all(col in df.columns for col in ['Context', 'Response']):
        print("Error: Dataset missing required columns ('Context', 'Response')")
        exit(1)

    df = df.dropna(subset=['Context', 'Response']).drop_duplicates()
    print(f"Cleaned Dataset Shape: {df.shape}")

    # Compute response type and crisis flag
    crisis_keywords = ['suicide', 'hopeless', 'worthless', 'kill', 'harm', 'desperate', 'overwhelmed', 'alone']
    df["response_type"] = df["Response"].apply(categorize_response)
    df["response_type_single"] = df["response_type"].apply(lambda x: x.split("|")[0])
    df["crisis_flag"] = df["Context"].apply(
        lambda x: sum(1 for word in crisis_keywords if word in str(x).lower()) > 0
    )

    # Initialize ChromaDB client
    try:
        if os.path.exists(chroma_db_path):
            print(f"Clearing existing ChromaDB at {chroma_db_path}")
            shutil.rmtree(chroma_db_path)
        os.makedirs(chroma_db_path, exist_ok=True)
        chroma_client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        print("Ensure ChromaDB version is compatible (e.g., 0.5.x) and no other processes are accessing the database.")
        exit(1)

    # Initialize OpenAI embeddings
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    except Exception as e:
        print(f"Error initializing OpenAI embeddings: {e}")
        exit(1)

    # Create or reset collection
    collection_name = "mental_health_conversations"
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}' if it existed")
    except:
        print(f"No existing collection '{collection_name}' to delete")
    try:
        collection = chroma_client.create_collection(name=collection_name)
        print(f"Created new collection '{collection_name}'")
    except Exception as e:
        print(f"Error creating Chroma collection: {e}")
        exit(1)

    # Prepare documents
    documents = df["Context"].tolist()
    metadatas = [
        {
            "response": row["Response"],
            "response_type": row["response_type_single"],
            "crisis_flag": bool(row["crisis_flag"])
        }
        for _, row in df.iterrows()
    ]
    ids = [f"doc_{i}" for i in range(len(documents))]

    # Generate embeddings and add to collection
    try:
        embeddings_vectors = embeddings.embed_documents(documents)
        collection.add(
            documents=documents,
            embeddings=embeddings_vectors,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Vector database created in {chroma_db_path} with {len(documents)} documents")
    except Exception as e:
        print(f"Error generating embeddings or adding to collection: {e}")
        exit(1)

if __name__ == "__main__":
    create_vector_db()