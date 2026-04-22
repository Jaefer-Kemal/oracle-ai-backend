import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

# Add backend to path
sys.path.append(os.getcwd())

from models import SessionLocal, Document, VectorEntry, init_db
from services import CohereService, ParserService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-script")

def ingest_local_docs():
    db = SessionLocal()
    cohere = CohereService()
    parser = ParserService()
    
    docs_dir = "doc"
    files = ["Oracle_Overview.md", "Oracle_Services.md", "Oracle_Contacts.md"]
    
    for filename in files:
        path = os.path.join(docs_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            continue
            
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        file_hash = parser.get_file_hash(content.encode())
        
        # Check if already exists
        if db.query(Document).filter(Document.file_hash == file_hash).first():
            logger.info(f"Skipping {filename} (duplicate)")
            continue
            
        logger.info(f"Ingesting {filename}...")
        chunks = parser.chunk_text(content)
        embeddings = cohere.get_embeddings(chunks)
        
        new_doc = Document(filename=filename, file_hash=file_hash, category="Oracle Docs", source_type="file")
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        
        for chunk, emb in zip(chunks, embeddings):
            entry = VectorEntry(doc_id=new_doc.id, content=chunk, embedding=emb, metadata_json={"category": "Oracle Docs"})
            db.add(entry)
        
        db.commit()
        logger.info(f"Successfully ingested {len(chunks)} chunks for {filename}.")

    db.close()

if __name__ == "__main__":
    init_db() # Ensure tables exist
    ingest_local_docs()
