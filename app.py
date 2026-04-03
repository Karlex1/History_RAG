import os
from dotenv import load_dotenv

import chainlit as cl
from google import genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

print("🚀 Chainlit app is starting...")

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment.")

VECTOR_DB_PATH = "history_vector_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
GENERATION_MODEL = "gemini-2.5-flash"

client = genai.Client(api_key=GEMINI_API_KEY)

embeddings = None
db = None
reranker = None

def load_rag_components():
    global embeddings, db, reranker

    if embeddings is None:
        print("📦 Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("✅ Embeddings model loaded")

    if db is None:
        print("📚 Loading FAISS vector DB...")
        if not os.path.exists(VECTOR_DB_PATH):
            raise FileNotFoundError(
                f"Vector DB folder '{VECTOR_DB_PATH}' not found. "
                "Upload it to the repo or rebuild it during deployment."
            )
        db = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ FAISS vector DB loaded")

    if reranker is None:
        print("🧠 Loading reranker model...")
        reranker = CrossEncoder(RERANKER_MODEL_NAME)
        print("✅ Reranker model loaded")

def get_source_info(doc):
    file = doc.metadata.get("source_file") or doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", None)

    if isinstance(file, str):
        file = file.split("\\")[-1].split("/")[-1]

    if page is not None:
        return file, str(page)
    return file, ""


def query_to_keyword(user_query: str) -> str:
    prompt = f"""
You are helping a history RAG system retrieve textbook passages for answers.

Rules:
- Return ONLY one line
- No bullet points
- No explanations
- No labels like "Core Topic"
- Just keywords separated by spaces

Example:
User: Nationalism in India 1919
Output: Indian nationalism 1919 Rowlatt Act Jallianwala Bagh Gandhi

Do not answer the question.
Only return the rewritten retrieval query.

User query:
{user_query}
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )
    return response.text.strip()


def retrieve_and_rerank(retrieval_query: str):
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 15}
    )

    results = retriever.invoke(retrieval_query)

    if not results:
        return [], []

    pairs = [(retrieval_query, doc.page_content) for doc in results]
    scores = reranker.predict(pairs)

    scored_results = list(zip(results, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)

    top_docs = [doc for doc, score in scored_results if score > 0.15][:5]
    if not top_docs:
        top_docs = [doc for doc, score in scored_results[:3]]

    return scored_results, top_docs


def build_context(top_docs):
    context = ""
    used_sources = []

    for doc in top_docs:
        source, page = get_source_info(doc)

        if page:
            source_label = f"{source}, page {page}"
            context += f"[Source: {source}, page {page}]\n{doc.page_content}\n\n"
        else:
            source_label = source
            context += f"[Source: {source}]\n{doc.page_content}\n\n"

        if source_label not in used_sources:
            used_sources.append(source_label)

    return context, used_sources


def answer_with_gemini(user_query: str, context: str) -> str:
    prompt = f"""
You are a Class 12 history teacher answering CBSE student questions.

Answer the question using only the given context.

Context:
{context}

Question:
{user_query}

Instructions:
- Ignore unrelated information
- Add source reference at the end of each point
- Use the exact source labels provided in the context
- Do NOT add information not in context
- If not found, say: "Answer not found in knowledge base"
- Write clear exam-style notes in bullet points
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )
    return response.text.strip()

@cl.on_chat_start
async def start():
    loading_msg = cl.Message(content="Initializing History RAG system...")
    await loading_msg.send()

    try:
        load_rag_components()

        await loading_msg.remove()
        await cl.Message(
            content=(
                "Hi! I am your Class 12 History RAG tutor.\n\n"
                "Ask me questions like:\n"
                "- Detailed notes on Maratha Empire\n"
                "- Nationalism in India 1919\n"
                "- Explain the Mughal Mansabdari system"
            )
        ).send()

    except Exception as e:
        await loading_msg.remove()
        await cl.Message(
            content=f"Startup error: {str(e)}"
        ).send()


@cl.on_message
async def main(message: cl.Message):
    user_query = message.content.strip()

    if not user_query:
        await cl.Message(content="Please enter a valid question.").send()
        return

    status = cl.Message(content="Processing your question...")
    await status.send()

    try:
        # Safe even if already loaded
        load_rag_components()

        retrieval_query = query_to_keyword(user_query)
        scored_results, top_docs = retrieve_and_rerank(retrieval_query)

        if not top_docs:
            await status.remove()
            await cl.Message(content="Answer not found in knowledge base").send()
            return

        context, used_sources = build_context(top_docs)
        final_answer = answer_with_gemini(user_query, context)

        sources_text = "**Sources Used:**\n"
        for source in used_sources:
            sources_text += f"- {source}\n"

        await status.remove()
        await cl.Message(content=final_answer).send()
        await cl.Message(content=sources_text, author="Sources").send()

    except Exception as e:
        await status.remove()
        await cl.Message(content=f"Error: {str(e)}").send()
