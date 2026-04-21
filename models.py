import os
import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, JSON, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv

logger = logging.getLogger("rag-backend")

load_dotenv()

# Database Setup (Neon/Postgres)
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,        # Test connection before use — kills stale Neon connections
    pool_recycle=300,          # Recycle connections every 5 min to avoid Neon idle drops
    pool_size=5,
    max_overflow=10,
    connect_args={"connect_timeout": 10},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_hash = Column(String, unique=True, index=True)
    category = Column(String, index=True, default="General")
    source_type = Column(String)  # 'file' or 'manual'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Soft Delete Fields
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)
    
    # Relationship to embeddings
    embeddings = relationship("VectorEntry", back_populates="document", cascade="all, delete-orphan")

class VectorEntry(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"))
    content = Column(Text)
    embedding = Column(Vector(1536))  # embed-v4.0 default dimension
    metadata_json = Column(JSON)

    document = relationship("Document", back_populates="embeddings")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True) # UUID or custom string from frontend
    title = Column(String, default="New Conversation")
    is_internal = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship("ChatHistory", back_populates="session", cascade="all, delete-orphan")

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=True)
    query = Column(Text)
    answer = Column(Text)
    context_used = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")

class AppSettings(Base):
    __tablename__ = "app_settings"

    key = Column(String, primary_key=True)
    value = Column(Text)

def init_db():
    from sqlalchemy import text, inspect
    
    # 1. Enable Vector Extension
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    
    # 2. Basic Table Creation
    Base.metadata.create_all(bind=engine)
    
    # 3. Self-Healing Migrations (Missing Column Check)
    inspector = inspect(engine)
    with engine.connect() as conn:
        # Check 'documents' table
        doc_cols = [c["name"] for c in inspector.get_columns("documents")]
        if "is_deleted" not in doc_cols:
            conn.execute(text("ALTER TABLE documents ADD COLUMN is_deleted BOOLEAN DEFAULT FALSE"))
            print("Migration: Added is_deleted to documents")
        if "deleted_at" not in doc_cols:
            conn.execute(text("ALTER TABLE documents ADD COLUMN deleted_at TIMESTAMP"))
            print("Migration: Added deleted_at to documents")
            
        # Check 'chat_history' table
        hist_cols = [c["name"] for c in inspector.get_columns("chat_history")]
        if "session_id" not in hist_cols:
            conn.execute(text("ALTER TABLE chat_history ADD COLUMN session_id VARCHAR REFERENCES chat_sessions(id) ON DELETE CASCADE"))
            print("Migration: Added session_id to chat_history")
            
        # Check 'chat_sessions' table
        sess_cols = [c["name"] for c in inspector.get_columns("chat_sessions")]
        if "is_internal" not in sess_cols:
            conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN is_internal BOOLEAN DEFAULT FALSE"))
            print("Migration: Added is_internal to chat_sessions")
        
        # Cleanup legacy Gemini cookies
        conn.execute(text("DELETE FROM app_settings WHERE key IN ('gemini_1psid', 'gemini_1psidts')"))
        print("Migration: Purged legacy Gemini cookies")
        
        # Check vector dimension — must match the current embedding model
        # embed-v4.0 = 1536 dims. If the column is wrong size, clear and recreate.
        try:
            result = conn.execute(text("""
                SELECT atttypmod 
                FROM pg_attribute 
                JOIN pg_class ON pg_attribute.attrelid = pg_class.oid
                WHERE pg_class.relname = 'embeddings' AND pg_attribute.attname = 'embedding'
            """)).fetchone()
            if result and result[0] != 1536:
                logger.warning(f"Vector dimension mismatch (found {result[0]}, need 1536). Dropping old embeddings...")
                conn.execute(text("DELETE FROM embeddings"))
                conn.execute(text("ALTER TABLE embeddings DROP COLUMN embedding"))
                conn.execute(text("ALTER TABLE embeddings ADD COLUMN embedding vector(1536)"))
                logger.info("Migration: Rebuilt embedding column to 1536 dimensions for embed-v4.0")
        except Exception as e:
            logger.warning(f"Could not check vector dimension: {e}")
        
        conn.commit()
    
    # 4. Pre-populate defaults and Initialize User
    db = SessionLocal()
    try:
        # User initialization from .env
        import os
        if not db.query(User).first():
            from auth import get_password_hash
            default_user = os.getenv("ADMIN_USERNAME", "admin")
            default_pass = os.getenv("ADMIN_PASSWORD", "admin123")
            db.add(User(username=default_user, hashed_password=get_password_hash(default_pass)))
            logger.info("Created default admin user from environment variables.")

        # Cleanup old AppSettings for auth
        stale_settings = db.query(AppSettings).filter(AppSettings.key.in_(["admin_username", "admin_password", "admin_password_hash"])).all()
        for s in stale_settings:
            db.delete(s)

        defaults = {
            "greeting_message": "Hello! I am your AI assistant. How can I help you today?",
            "fallback_message": "I'm sorry, I couldn't find specific information about that in my database. Please contact our support team at support@company.com for further assistance.",
            "similarity_threshold": "0.5",
            "grok_model": "grok-3-auto",
            "active_provider": "grok",
            "fallback_chain": '["grok", "gemini", "g4f"]',
            "suggested_questions": '["What topics are covered in the knowledge base?", "How can I contact support?", "What are your business hours?", "How do I get started?", "What services do you offer?"]'
        }
        for k, v in defaults.items():
            if not db.query(AppSettings).filter(AppSettings.key == k).first():
                db.add(AppSettings(key=k, value=v))
        db.commit()
    finally:
        db.close()
