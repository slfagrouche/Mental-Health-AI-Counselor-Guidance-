# api_mental_health.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from openai import OpenAI
import os
from dotenv import load_dotenv
from langsmith import Client, traceable
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize FastAPI app
app = FastAPI(title="Mental Health Counselor API")

# Initialize components
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()
output_dir = "mental_health_model_artifacts"

# Global variables for models and vector store
response_clf = None
crisis_clf = None
vectorizer = None
le = None
selector = None
lda = None
vector_store = None
llm = None
openai_client = None
langsmith_client = None

# Load models and initialize ChromaDB at startup
@app.on_event("startup")
async def startup_event():
    global response_clf, crisis_clf, vectorizer, le, selector, lda, vector_store, llm, openai_client, langsmith_client
    
    # Check environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set in .env file")
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in .env file")
    if not os.environ.get("LANGCHAIN_API_KEY"):
        logger.error("LANGCHAIN_API_KEY not set in .env file")
        raise HTTPException(status_code=500, detail="LANGCHAIN_API_KEY not set in .env file")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "MentalHealthCounselorPOC"

    # Initialize LangSmith client
    logger.info("Initializing LangSmith client")
    langsmith_client = Client()

    # Load saved components
    logger.info("Loading model artifacts")
    try:
        response_clf = joblib.load(f"{output_dir}/response_type_classifier.pkl")
        crisis_clf = joblib.load(f"{output_dir}/crisis_classifier.pkl")
        vectorizer = joblib.load(f"{output_dir}/tfidf_vectorizer.pkl")
        le = joblib.load(f"{output_dir}/label_encoder.pkl")
        selector = joblib.load(f"{output_dir}/feature_selector.pkl")
        
        try:
            lda = joblib.load(f"{output_dir}/lda_model.pkl")
        except Exception as lda_error:
            logger.warning(f"Failed to load LDA model: {lda_error}. Creating placeholder model.")
            from sklearn.decomposition import LatentDirichletAllocation
            lda = LatentDirichletAllocation(n_components=10, random_state=42)
            # Note: Placeholder is untrained; retrain for accurate results
            
    except FileNotFoundError as e:
        logger.error(f"Missing model artifact: {e}")
        raise HTTPException(status_code=500, detail=f"Missing model artifact: {e}")

    # Initialize ChromaDB
    chroma_db_path = f"{output_dir}/chroma_db"
    if not os.path.exists(chroma_db_path):
        logger.error(f"ChromaDB not found at {chroma_db_path}. Run create_vector_db.py first.")
        raise HTTPException(status_code=500, detail=f"ChromaDB not found at {chroma_db_path}. Run create_vector_db.py first.")
    
    try:
        logger.info("Initializing ChromaDB")
        chroma_client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=os.environ["OPENAI_API_KEY"],
            disallowed_special=(), 
            chunk_size=1000  
        )
        global vector_store
        vector_store = Chroma(
            client=chroma_client,
            collection_name="mental_health_conversations",
            embedding_function=embeddings
        )
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing ChromaDB: {e}")

    # Initialize OpenAI client and LLM
    logger.info("Initializing OpenAI client and LLM")
    global openai_client, llm
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.environ["OPENAI_API_KEY"]
    )

# Pydantic model for request
class PatientContext(BaseModel):
    context: str

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

# Feature engineering function
@traceable(run_type="tool", name="Engineer Features")
def engineer_features(context, response=""):
    context_clean = clean_text(context)
    context_len = len(context_clean.split())
    context_vader = analyzer.polarity_scores(context)['compound']
    context_questions = context.count('?')
    crisis_keywords = ['suicide', 'hopeless', 'worthless', 'kill', 'harm', 'desperate', 'overwhelmed', 'alone']
    context_crisis_score = sum(1 for word in crisis_keywords if word in context.lower())
    
    context_tfidf = vectorizer.transform([context_clean]).toarray()
    tfidf_cols = [f"tfidf_context_{i}" for i in range(context_tfidf.shape[1])]
    response_tfidf = np.zeros_like(context_tfidf)
    
    lda_topics = lda.transform(context_tfidf)
    
    feature_cols = ["context_len", "context_vader", "context_questions", "crisis_flag"] + \
                   [f"topic_{i}" for i in range(10)] + tfidf_cols + \
                   [f"tfidf_response_{i}" for i in range(response_tfidf.shape[1])]
    
    features = pd.DataFrame({
        "context_len": [context_len],
        "context_vader": [context_vader],
        "context_questions": [context_questions],
        **{f"topic_{i}": [lda_topics[0][i]] for i in range(10)},
        **{f"tfidf_context_{i}": [context_tfidf[0][i]] for i in range(context_tfidf.shape[1])},
        **{f"tfidf_response_{i}": [response_tfidf[0][i]] for i in range(response_tfidf.shape[1])},
    })
    
    crisis_features = features[["context_len", "context_vader", "context_questions"] + [f"topic_{i}" for i in range(10)]]
    crisis_flag = crisis_clf.predict(crisis_features)[0]
    if context_crisis_score > 0:
        crisis_flag = 1
    features["crisis_flag"] = crisis_flag
    
    return features, feature_cols

# Prediction function
@traceable(run_type="chain", name="Predict Response Type")
def predict_response_type(context):
    features, feature_cols = engineer_features(context)
    selected_features = selector.transform(features[feature_cols])
    pred_encoded = response_clf.predict(selected_features)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    confidence = response_clf.predict_proba(selected_features)[0].max()
    
    if "?" in context and context.count("?") > 0:
        pred_label = "Question"
    if "trying" in context.lower() and "hard" in context.lower() and not any(kw in context.lower() for kw in ["how", "what", "help"]):
        pred_label = "Validation"
    if "trying" in context.lower() and "positive" in context.lower() and not any(kw in context.lower() for kw in ["how", "what", "help"]):
        pred_label = "Question"
    
    crisis_flag = bool(features["crisis_flag"].iloc[0])
    
    return {
        "response_type": pred_label,
        "crisis_flag": crisis_flag,
        "confidence": confidence,
        "features": features.to_dict()
    }

# RAG suggestion function
@traceable(run_type="chain", name="RAG Suggestion")
def generate_suggestion_rag(context, response_type, crisis_flag):
    results = vector_store.similarity_search_with_score(context, k=3)
    retrieved_contexts = [
        f"Patient: {res[0].page_content}\nCounselor: {res[0].metadata['response']} (Type: {res[0].metadata['response_type']}, Crisis: {res[0].metadata['crisis_flag']}, Score: {res[1]:.2f})"
        for res in results
    ]
    
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an expert mental health counseling assistant. A counselor has provided the following patient situation:
        
        Patient Situation: {context}
        
        Predicted Response Type: {response_type}
        Crisis Flag: {crisis_flag}
        
        Based on the predicted response type and crisis flag, provide a suggested response for the counselor to use with the patient. The response should align with the response type ({response_type}) and be sensitive to the crisis level.
        
        For reference, here are similar cases from past conversations:
        {retrieved_contexts}
        
        Guidelines:
        - If Crisis Flag is True, prioritize safety, empathy, and suggest immediate resources (e.g., National Suicide Prevention Lifeline at 988).
        - For 'Empathetic Listening', focus on validating feelings without giving direct advice or questions.
        - For 'Advice', provide practical, actionable suggestions.
        - For 'Question', pose an open-ended question to encourage further discussion.
        - For 'Validation', affirm the patient's efforts or feelings.
        
        Output in the following format:
        ```json
        {{
            "suggested_response": "Your suggested response here",
            "risk_level": "Low/Moderate/High"
        }}
        ```
        """
    )
    
    rag_chain = (
        {
            "context": RunnablePassthrough(),
            "response_type": lambda x: response_type,
            "crisis_flag": lambda x: "Crisis" if crisis_flag else "No Crisis",
            "retrieved_contexts": lambda x: "\n".join(retrieved_contexts)
        }
        | prompt_template
        | llm
    )
    
    try:
        response = rag_chain.invoke(context)
        return eval(response.content.strip("```json\n").strip("\n```"))
    except Exception as e:
        logger.error(f"Error generating RAG suggestion: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating RAG suggestion: {str(e)}")

# Direct suggestion function
@traceable(run_type="chain", name="Direct Suggestion")
def generate_suggestion_direct(context, response_type, crisis_flag):
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an expert mental health counseling assistant. A counselor has provided the following patient situation:
        
        Patient Situation: {context}
        
        Predicted Response Type: {response_type}
        Crisis Flag: {crisis_flag}
        
        Provide a suggested response for the counselor to use with the patient, aligned with the response type ({response_type}) and sensitive to the crisis level.
        
        Guidelines:
        - If Crisis Flag is True, prioritize safety, empathy, and suggest immediate resources (e.g., National Suicide Prevention Lifeline at 988).
        - For 'Empathetic Listening', focus on validating feelings without giving direct advice or questions.
        - For 'Advice', provide practical, actionable suggestions.
        - For 'Question', pose an open-ended question to encourage further discussion.
        - For 'Validation', affirm the patient's efforts or feelings.
        - Strictly adhere to the response type. For 'Empathetic Listening', do not include questions or advice.
        
        Output in the following format:
        ```json
        {{
            "suggested_response": "Your suggested response here",
            "risk_level": "Low/Moderate/High"
        }}
        ```
        """
    )
    
    direct_chain = (
        {
            "context": RunnablePassthrough(),
            "response_type": lambda x: response_type,
            "crisis_flag": lambda x: "Crisis" if crisis_flag else "No Crisis"
        }
        | prompt_template
        | llm
    )
    
    try:
        response = direct_chain.invoke(context)
        return eval(response.content.strip("```json\n").strip("\n```"))
    except Exception as e:
        logger.error(f"Error generating direct suggestion: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating direct suggestion: {str(e)}")

# API Endpoints
@app.post("/suggest")
async def get_suggestion(context: PatientContext):
    logger.info(f"Received suggestion request for context: {context.context}")
    prediction = predict_response_type(context.context)
    suggestion_rag = generate_suggestion_rag(context.context, prediction["response_type"], prediction["crisis_flag"])
    suggestion_direct = generate_suggestion_direct(context.context, prediction["response_type"], prediction["crisis_flag"])
    
    return {
        "context": context.context,
        "response_type": prediction["response_type"],
        "crisis_flag": prediction["crisis_flag"],
        "confidence": prediction["confidence"],
        "rag_suggestion": suggestion_rag["suggested_response"],
        "rag_risk_level": suggestion_rag["risk_level"],
        "direct_suggestion": suggestion_direct["suggested_response"],
        "direct_risk_level": suggestion_direct["risk_level"]
    }

@app.get("/health")
async def health_check():
    if all([response_clf, crisis_clf, vectorizer, le, selector, lda, vector_store, llm]):
        return {"status": "healthy", "message": "All models and vector store loaded successfully"}
    logger.error("Health check failed: One or more components not loaded")
    raise HTTPException(status_code=500, detail="One or more components failed to load")

@app.get("/metadata")
async def get_metadata():
    try:
        collection = vector_store._client.get_collection("mental_health_conversations")
        count = collection.count()
        return {"collection_name": "mental_health_conversations", "document_count": count}
    except Exception as e:
        logger.error(f"Error retrieving metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving metadata: {str(e)}")
