from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_script import Pinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set API and Index
# PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_KEY = "pcsk_4TUYdy_FZFVo2qAsSuwsFf7V7XVPydKshpurvHbpvaBm9HsZTaA33jcE2Bdrc1gevFq5gY"
INDEX_NAME = "shypbuddy-vector"

# Define functions first
def load_pdf(data_path):
    """Extract data from the PDF"""
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    """Create text chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    """Download embedding model"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def main():
    try:
        # 1. Load and chunk text
        print("📄 Loading PDF documents...")
        documents = load_pdf("data/")
        print(f"✅ Loaded {len(documents)} documents")
        
        print("✂️ Splitting text into chunks...")
        text_chunks = text_split(documents)
        texts = [chunk.page_content for chunk in text_chunks]
        print(f"✅ Created {len(text_chunks)} text chunks")

        # 2. Get embeddings
        print("🤖 Loading embedding model...")
        embedding_model = download_hugging_face_embeddings()
        print("📊 Generating embeddings...")
        embeddings = embedding_model.embed_documents(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")

        # 3. Connect to Pinecone
        print("🌲 Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        print(f"✅ Connected to Pinecone index '{INDEX_NAME}'")

        # 4. Create vectors and upsert
        print("📤 Preparing vectors for upload...")
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

        print("🚀 Uploading vectors to Pinecone...")
        # Upload in batches for better performance
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"📤 Uploaded batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")

        print(f"✅ Successfully uploaded {len(vectors)} vectors to Pinecone index '{INDEX_NAME}'")
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"📊 Index stats: {stats}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()