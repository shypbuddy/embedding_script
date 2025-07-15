from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# Pinecone setup
PINECONE_API_KEY = "pcsk_4TUYdy_FZFVo2qAsSuwsFf7V7XVPydKshpurvHbpvaBm9HsZTaA33jcE2Bdrc1gevFq5gY"
INDEX_NAME = "shypbuddy-vector"


# Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


# -------------------------------
# Custom text splitting function
# -------------------------------
def text_split(documents, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    all_chunks = []
    for doc in documents:
        chunks = splitter.create_documents([doc.page_content])
        # Optional: add page number/source if available
        for chunk in chunks:
            chunk.metadata = {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown")
            }
            all_chunks.append(chunk)
    return all_chunks

# -------------------------------
# 1. Load PDF documents
# -------------------------------
documents = load_pdf("data/")
print(f"✅ Loaded {len(documents)} PDF pages")

# -------------------------------
# 2. Split text into chunks
# -------------------------------
text_chunks = text_split(documents, chunk_size=800, chunk_overlap=100)
texts = [chunk.page_content for chunk in text_chunks]
print(f"✂️ Split into {len(text_chunks)} text chunks")

# -------------------------------
# 3. Generate embeddings
# -------------------------------
embedding_model = download_hugging_face_embeddings()
embeddings = embedding_model.embed_documents(texts)
print("🧠 Embeddings generated")

# -------------------------------
# 4. Connect to Pinecone
# -------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# -------------------------------
# 4.1 Delete all existing vectors in the index
# -------------------------------
print(f"🗑️ Deleting all existing vectors in index '{INDEX_NAME}'...")
index.delete(delete_all=True)
print(f"✅ All existing vectors deleted from index '{INDEX_NAME}'")

# -------------------------------
# 5. Upsert into Pinecone
# -------------------------------
vectors = [
    {
        "id": f"vec-{i}",
        "values": embeddings[i],
        "metadata": {
            "text": texts[i],
            "source": text_chunks[i].metadata.get("source", "unknown"),
            "page": text_chunks[i].metadata.get("page", "unknown")
        }
    }
    for i in range(len(texts))
]

index.upsert(vectors=vectors)
print(f"✅ Uploaded {len(vectors)} vectors to Pinecone index '{INDEX_NAME}'")
