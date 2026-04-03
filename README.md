# 📘 History RAG System (Class 12 NCERT)

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG) system** for answering Class 12 History questions using NCERT textbooks.

Unlike traditional LLM-based systems that rely only on pre-trained knowledge, this solution combines **semantic search with large language models (LLMs)** to generate answers grounded in real textbook content. This significantly improves **accuracy, reliability, and reduces hallucination**, making it suitable for educational and knowledge-based applications.

---

## 🧠 Key Features

- 📄 Supports multiple document formats (**PDF, DOCX**)  
- 🔍 Fast and accurate **semantic search using FAISS**  
- 🧠 Improved relevance with **CrossEncoder-based reranking**  
- 🤖 Context-aware answer generation using **Gemini LLM**  
- ❌ Hallucination control via **strict prompt engineering**  
- 💬 Interactive conversational UI built using **Chainlit**  
- ⚙️ Modular and scalable architecture for easy extension  

---

## ⚙️ Tech Stack

- **Embeddings:** BAAI/bge-base-en-v1.5 
- **Vector Database:** FAISS  
- **Reranker:** bge-reranker-base (CrossEncoder)  
- **LLM:** Gemini  
- **Frameworks:** LangChain, SentenceTransformers  
- **UI:** Chainlit  

---

## 🔄 How It Works

The system follows an end-to-end RAG pipeline:

1. Load documents from PDFs and DOCX files  
2. Clean and preprocess text to remove noise  
3. Split content into chunks (size: 800, overlap: 150)  
4. Convert chunks into embeddings  
5. Store embeddings in FAISS for efficient retrieval  
6. Retrieve relevant chunks using semantic search (MMR)  
7. Rerank results using CrossEncoder for higher accuracy  
8. Generate final answers using LLM based on retrieved context  

---

## 🏗️ System Architecture

#### 📚 Knowledge Base Preparation (Offline Phase)

Loading Data  
→ Data Cleaning & Preprocessing  
→ Text Chunking (chunk_size=800, overlap=150)  
→ Embedding Initialization & Encoding  
→ Vector Database (FAISS Index Storage)

---

#### 💬 Query Processing (Online Phase)

User Query  
→ Query Embedding  
→ FAISS Similarity Search  
→ Top-K Retrieval  
→ Reranking (CrossEncoder)  
→ Context Filtering  
→ LLM (Gemini)  
→ Final Answer

---

## 🖥️ User Interface

The current system uses **Chainlit** to provide a conversational interface for interacting with the RAG pipeline. This allows users to ask questions in real-time and receive context-aware responses.

Chainlit is highly effective for rapid prototyping and debugging LLM workflows.

For future improvements, the UI can be enhanced by transitioning to a **React-based frontend**, which would provide:

- Better user experience and responsive design  
- Structured layouts (dashboard, history, sources)  
- User authentication and session handling  
- Scalable frontend-backend integration for production use  

---

## 📌 Example Use Case

**Question:** *What were the causes of the Revolt of 1857?*

The system:
- Searches relevant sections from NCERT textbooks  
- Filters high-quality context using reranking  
- Generates a structured, context-based answer  

---

## 🧪 What I Learned (Key Insights)

### 🔹 Embedding Models Matter

One of the most critical learnings was the importance of **embedding consistency**.

Both documents and queries must use the **same embedding model** to ensure they exist in the same vector space. Mismatched embeddings lead to poor retrieval results.

I also explored trade-offs between:
- *MiniLM* → faster but less precise  
- *bge-base* → slower but more semantically accurate  

---

### 🔹 Retrieval vs Reranking

Initial retrieval using FAISS alone often returned **partially relevant or noisy results**.

To address this, I implemented a **CrossEncoder reranker**, which evaluates query-document pairs together and ranks them based on true relevance.

This significantly improved:
- Context quality  
- Answer accuracy  
- Reduction of irrelevant information  

---

### 🔹 LLM vs RAG (Core Understanding)

This project clarified a key concept:

- **LLM (Standalone):**  
  Generates answers from training data → may hallucinate  

- **RAG System:**  
  Retrieves real data first → then generates answer  

👉 In simple terms:  
**LLM generates language, RAG provides knowledge**

---

### 🔹 Hallucination Control

To ensure trustworthy outputs:
- Used strict prompt instructions to restrict answers to context  
- Implemented fallback response when data is not found  

This prevents misleading or fabricated answers.

---

### 🔹 Chunking Strategy

Chunking plays a crucial role in performance:

- Smaller chunks → better precision  
- Larger chunks → better context  

An optimal balance (500 size, 100 overlap) gave the best results.

---

### 🔹 End-to-End System Design

This project helped build a strong understanding of:
- Vector databases (FAISS)  
- Semantic search systems  
- Retrieval pipelines  
- LLM integration  

---

## 🔮 Future Improvements

- 🌐 Upgrade UI from Chainlit to a **React-based frontend**  
- 📖 Add **source citations for transparency**  
- 🔍 Implement **hybrid search (BM25 + embeddings)**  
- 🌍 Enable **multilingual support**  
- 🔁 Add feedback loop for continuous improvement  

---

## 📎 Author

**Sanchit**  
B.Tech IT | Full Stack & AI/ML Developer  

- GitHub: https://github.com/Karlex1  
- LinkedIn: https://www.linkedin.com/in/sanchit-312928214/

---

## ⭐ Keywords

Retrieval Augmented Generation, RAG system, FAISS vector database, semantic search, LLM project, Gemini AI, LangChain project, AI knowledge base, document question answering system, embedding models, CrossEncoder reranker, NLP project, AI for education, History_RAG + karlex1, Hist rag + karlex1
