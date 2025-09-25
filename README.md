# database-reader-# db_demo.py
import os
import sqlite3
import pandas as pd
import faiss
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Setup
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing! Add it to your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Ecommerce Database API with RAG")

DB_PATH = "ecommerce.db"
INDEX_PATH = "vector.index"

# Embedding model (local, light)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Models
class QueryRequest(BaseModel):
    question: str


# Utility Functions
def get_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("Database not found! Please upload a CSV first.")
    return sqlite3.connect(DB_PATH)


def build_faiss_index():
    """Build FAISS index from ecommerce DB"""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM ecommerce", conn)
    conn.close()

    # Convert all rows to text chunks
    docs = df.astype(str).agg(" | ".join, axis=1).tolist()

    embeddings = embedder.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    return docs, index


def load_index(docs_cache=None):
    if not os.path.exists(INDEX_PATH):
        return build_faiss_index()
    index = faiss.read_index(INDEX_PATH)
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM ecommerce", conn)
    conn.close()
    docs = df.astype(str).agg(" | ".join, axis=1).tolist()
    return docs, index


def rag_answer(question: str):
    docs, index = load_index()
    q_emb = embedder.encode([question], convert_to_numpy=True)

    D, I = index.search(q_emb, k=3)  # top 3 matches
    retrieved = [docs[i] for i in I[0] if i < len(docs)]

    context = "\n".join(retrieved)

    prompt = f"""
    You are an assistant for an ecommerce database.
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# Route
@app.get("/")
def root():
    return {"status": "ok", "msg": "Ecommerce API with RAG is running!"}


@app.post("/load_csv")
async def load_csv(file: UploadFile = File(...)):
    """Upload a CSV file and load into SQLite + build FAISS index"""
    try:
        df = pd.read_csv(file.file)
        conn = sqlite3.connect(DB_PATH)
        df.to_sql("ecommerce", conn, if_exists="replace", index=False)
        conn.close()

        # build vector index
        build_faiss_index()

        return {"status": "success", "rows_loaded": len(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading CSV: {str(e)}")


@app.post("/query_sql")
async def query_db(req: QueryRequest):
    """Execute SQL queries directly"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(req.question)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        results = [dict(zip(columns, row)) for row in rows]
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error executing query: {str(e)}")


@app.post("/query_nl")
async def query_nl(req: QueryRequest):
    """Ask natural language questions (RAG)"""
    try:
        answer = rag_answer(req.question)
        return {"status": "success", "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"RAG Error: {str(e)}")
