import os
import re
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Loading=> cleaning=>chuncking=> embedding=>storing in db
def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("---", " ")
    text = re.sub(r"THEMES IN INDIAN HISTORY\s*[–-]?\s*PART\s*[IVX]+\d*", " ", text)
    text = re.sub(r"Reprint\s+\d{4}-\d{2}", " ", text)
    text = re.sub(r"Fig\.\s*\d+(\.\d+)?", " ", text)
    text = re.sub(r"How was .*?\?", " ", text)
    text = re.sub(r"Why were .*?\?", " ", text)
    text = re.sub(r"What do .*?\?", " ", text)
    text = re.sub(r"Read any two .*?\.", " ", text)
    text = re.sub(r"Discuss\.\.\..*?\?", " ", text)
    text = re.sub(r"ANSWER IN.*", " ", text)
    text = re.sub(r"\d+\.\s.*?\?", " ", text)
    text = re.sub(r"[A-Z][a-z]+ [A-Z][a-z]+\. \d{4}\. .*?\.", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

#loader step 1
def load_documents(pdf_folder: str, docx_path: str):
    docs = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_folder, file)
            loader = PyPDFLoader(path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source_file"] = file
            docs.extend(loaded_docs)
    docx_loader = Docx2txtLoader(docx_path)
    docx_docs = docx_loader.load()

    for doc in docx_docs:
        doc.metadata["source_file"] = os.path.basename(docx_path)
    docs.extend(docx_docs)
    return docs
#we have the docs into pdf folder we r taking it and putting it into our 
#loader = PyPDFLoader(path)
          #  loaded_docs = loader.load()
          
          # at step 1 we extracted text text from the pdf using loader = PyPDFLoader(path)  loaded_docs = loader.load() and in output we have list of document object i.e; list of pages content and eachones metadata 

def build_vector_db():
    pdf_folder = "./NCERT_Class12"
    docx_path = "./Research Completed, Ready for Questions.docx"
    docs = load_documents(pdf_folder, docx_path)
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    filtered_chunks = [c for c in chunks if len(c.page_content.split()) > 40]
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )
    vector_db = FAISS.from_documents(filtered_chunks, embeddings) # indexing 
    # Embedding model during indexing and quering should be same 
    vector_db.save_local("history_vector_db")
    print(f"Saved vector DB with {len(filtered_chunks)} chunks.")


if __name__ == "__main__":
    build_vector_db()