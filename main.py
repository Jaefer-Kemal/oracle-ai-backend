import asyncio
import logging
import sys
import uuid
import threading
import os
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import or_, func, desc
from pydantic import BaseModel
import time

from models import SessionLocal, init_db, Document, VectorEntry, ChatHistory, AppSettings, ChatSession, User
from services import CohereService, AIService, ParserService
from core.providers.factory import ProviderFactory
from auth import create_access_token, create_refresh_token, decode_access_token, decode_refresh_token, get_password_hash, verify_password, get_current_user

# --- Professional Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("rag-backend")

# --- API Metadata ---
tags_metadata = [
    {"name": "General", "description": "System status and public configuration."},
    {"name": "Ingestion", "description": "Upload and process files or manual Q&A entries."},
    {"name": "Query", "description": "Context-aware RAG search and AI generation."},
    {"name": "Admin", "description": "Manage documents, trash, chat history, and system settings."},
    {"name": "Auth", "description": "Authentication and user management."},
]

app = FastAPI(
    title="Professional RAG API",
    description="Corporate RAG system with Sessions, Soft-Delete, and Grok-powered intelligence.",
    version="3.0.0",
    openapi_tags=tags_metadata
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Error Handling ---
@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "An internal server error occurred.", "detail": str(exc)}
    )

# Initialize Services
cohere_service = CohereService()
ai_service = AIService()
parser_service = ParserService()
ingestion_lock = threading.Lock()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Schemas ---
class QueryRequest(BaseModel):
    input: str
    session_id: Optional[str] = None
    is_internal: Optional[bool] = False
    history: Optional[List[Dict[str, str]]] = None # List of {"q": "...", "a": "..."}

class ManualKnowledgeRequest(BaseModel):
    title: str
    question: str
    answer: str
    category: Optional[str] = "Manual Entry"

class ConfigUpdate(BaseModel):
    greeting_message: Optional[str] = None
    fallback_message: Optional[str] = None
    similarity_threshold: Optional[str] = None
    suggested_questions: Optional[str] = None
    grok_model: Optional[str] = None
    active_provider: Optional[str] = None
    fallback_chain: Optional[str] = None
    gemini_1psid: Optional[str] = None
    gemini_1psidts: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

class BulkDeleteRequest(BaseModel):
    ids: List[__import__('typing').Union[str, int]]

class StandardResponse(BaseModel):
    status: str
    message: str
    success: Optional[bool] = None
    chunks: Optional[int] = None
    data: Optional[dict] = None

class ProfileUpdate(BaseModel):
    display_name: Optional[str] = None
    new_password: Optional[str] = None
    current_password: str

# --- Lifecycle ---
@app.on_event("startup")
def on_startup():
    logger.info("Initializing Database...")
    init_db()
    
    # Initialize AI Config Cache
    db = SessionLocal()
    try:
        ProviderFactory.reload_config(db)
    finally:
        db.close()
        
    logger.info("Backend Ready.")

# --- Public Endpoints ---

@app.get("/", tags=["General"], summary="Check API Status")
def read_root():
    return {"status": "online", "message": "Pro RAG API v3.0 Active."}

@app.get("/config/greeting", tags=["General"], summary="Fetch Greeting Message")
def get_greeting(db: Session = Depends(get_db)):
    setting = db.query(AppSettings).filter(AppSettings.key == "greeting_message").first()
    return {"greeting": setting.value if setting else "Hello!"}

@app.get("/config/suggestions", tags=["General"], summary="Fetch Suggested Questions")
def get_suggestions(db: Session = Depends(get_db)):
    setting = db.query(AppSettings).filter(AppSettings.key == "suggested_questions").first()
    import json
    try:
        return {"suggestions": json.loads(setting.value) if setting else []}
    except Exception:
        return {"suggestions": []}

# --- Ingestion Endpoints ---

@app.post("/ingest", tags=["Ingestion"], response_model=StandardResponse, summary="Upload a File")
def ingest_file(
    file: UploadFile = File(...),
    category: str = Form("General"),
    db: Session = Depends(get_db),
    user: str = Depends(get_current_user)
):
    with ingestion_lock:
        content = file.file.read()
        file_hash = parser_service.get_file_hash(content)

        existing = db.query(Document).filter(Document.file_hash == file_hash).first()
        if existing:
            if existing.is_deleted: # Restore if was in trash
                existing.is_deleted = False
                existing.deleted_at = None
                db.commit()
                return {"status": "restored", "message": "Document restored from trash."}
            return {"status": "skipped", "success": False, "message": "Document already exists (duplicate hash). No changes made.", "chunks": 0}

        ext = file.filename.split(".")[-1].lower()
        try:
            if ext == "pdf": raw_text = parser_service.extract_text_from_pdf(content)
            elif ext == "docx": raw_text = parser_service.extract_text_from_docx(content)
            elif ext == "md": raw_text = content.decode("utf-8")
            else: raw_text = content.decode("utf-8")
        except Exception as e:
            raise HTTPException(status_code=400, detail="File parsing failed.")

        # Auto-Categorization Logic
        final_category = category
        if category == "General" or not category:
            try:
                final_category = ai_service.suggest_category(raw_text, db)
                logger.info(f"AI suggested category for '{file.filename}': {final_category}")
            except Exception as e:
                logger.warning(f"Failed to auto-categorize: {e}")
                final_category = "General"

        chunks = parser_service.chunk_text(raw_text)
        embeddings = cohere_service.get_embeddings(chunks)

        new_doc = Document(filename=file.filename, file_hash=file_hash, category=final_category, source_type="file")
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)

        # Batch commits to prevent packet overload / connection drops with huge 1536-dim vectors
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            entry = VectorEntry(doc_id=new_doc.id, content=chunk, embedding=emb, metadata_json={"category": final_category})
            db.add(entry)
            
            if (idx + 1) % 50 == 0:
                try:
                    db.commit()
                except Exception as e:
                    db.rollback()
                    logger.error(f"Error during batched insert at {idx+1}: {e}")
                    raise HTTPException(status_code=500, detail="Database insertion failed.")

        try:
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail="Database final insertion failed.")
        return {"status": "success", "success": True, "message": f"Ingested {len(chunks)} chunks from '{file.filename}'.", "chunks": len(chunks)}

@app.post("/ingest/manual", tags=["Ingestion"], response_model=StandardResponse, summary="Add Manual Q&A")
def ingest_manual(req: ManualKnowledgeRequest, db: Session = Depends(get_db), user: str = Depends(get_current_user)):
    with ingestion_lock:
        combined_text = f"Question: {req.question}\nAnswer: {req.answer}"
        manual_hash = parser_service.get_file_hash(combined_text.encode())
        
        existing = db.query(Document).filter(Document.file_hash == manual_hash).first()
        if existing:
            if existing.is_deleted:
                existing.is_deleted = False
                db.commit()
                return {"status": "restored", "success": True, "message": "Entry was in trash and has been restored.", "chunks": 1}
            return {"status": "skipped", "success": False, "message": "An identical entry already exists in the knowledge base.", "chunks": 0}

        embeddings = cohere_service.get_embeddings([combined_text])
        new_doc = Document(filename=req.title, file_hash=manual_hash, category=req.category, source_type="manual")
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)

        entry = VectorEntry(doc_id=new_doc.id, content=combined_text, embedding=embeddings[0], metadata_json={"title": req.title})
        db.add(entry)
        db.commit()
        return {"status": "success", "success": True, "message": f"Manual entry '{req.title}' added to knowledge base.", "chunks": 1}

@app.post("/ingest/check", tags=["Ingestion"], summary="Check for similar facts")
def check_synergy(req: ManualKnowledgeRequest, db: Session = Depends(get_db), user: str = Depends(get_current_user)):
    """Check if a similar fact already exists to prevent duplicate ingestion."""
    combined_text = f"Question: {req.question}\nAnswer: {req.answer}"
    query_emb = cohere_service.get_embeddings([combined_text])[0]
    
    results = db.query(VectorEntry, VectorEntry.embedding.cosine_distance(query_emb).label("dist"))\
        .join(Document)\
        .filter(Document.is_deleted == False)\
        .order_by("dist").limit(3).all()
        
    suggestions = []
    for r in results:
        if r.dist < 0.4: # Synergy threshold
            suggestions.append({
                "content": r.VectorEntry.content,
                "score": round(1 - r.dist, 3)
            })
            
    return {"suggestions": suggestions}

# --- Query Endpoints ---

@app.post("/query", tags=["Query"], summary="Ask a Question")
async def query_rag(req: QueryRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    
    # Pulled from Factory Cache vs DB Query
    # NOTE: 'similarity_threshold' is stored as a DISTANCE value (0.0=perfect, 1.0=no match)
    # So a threshold of 0.85 means: include results with distance <= 0.85 (similarity >= 0.15)
    threshold = float(ProviderFactory._config_cache.get("similarity_threshold", "0.85"))

    # Session Management
    session_id = req.session_id or str(uuid.uuid4())
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    
    if not session:
        session = ChatSession(id=session_id, title="New Conversation", is_internal=req.is_internal)
        db.add(session)
        db.commit()

    # Get History — prefer history passed in request (for stateless iframe), otherwise check DB
    if req.history:
        history_data = req.history[-10:]
    else:
        history_entries = db.query(ChatHistory).filter(ChatHistory.session_id == session_id).order_by(desc(ChatHistory.timestamp)).limit(10).all()
        history_data = [{"q": h.query, "a": h.answer} for h in reversed(history_entries)]

    # 1. Query Reformulation — runs on ALL queries (not just history-backed ones)
    # This strips personal noise ("my name is X") and extracts core intent
    # 1. Query Reformulation (Threaded)
    history_data_list = history_data # local ref
    search_query = await asyncio.to_thread(ai_service.decontextualize_query, req.input, history_data_list, db)

    # 2. Retrieval (Threaded - Safe)
    resp = await asyncio.to_thread(cohere_service.get_embeddings, [search_query], input_type="search_query")
    query_emb = resp[0]
    
    # Run the query in main thread or use a robust thread-safe pattern
    # We'll use the main thread for the actual query to ensure session stability
    results = db.query(VectorEntry, VectorEntry.embedding.cosine_distance(query_emb).label("dist"))\
        .join(Document)\
        .filter(Document.is_deleted == False)\
        .order_by("dist").limit(5).all()
    
    # Corrected Logic: Threshold is a DISTANCE value.
    # Lower distance = more similar. Filter: include if dist <= threshold.
    valid_results = [r for r in results if r.dist <= threshold]
    
    # Debug: Log what we found
    logger.info(f"Vector Search: {len(results)} raw results, {len(valid_results)} passed threshold ({threshold}) for query: '{search_query[:60]}'")
    for r in results:
        logger.info(f"  -> Doc: {r.VectorEntry.document.filename if r.VectorEntry.document else 'N/A'} | dist={r.dist:.4f} | similarity={round((1-r.dist)*100, 1)}% | passed={'YES' if r.dist <= threshold else 'NO'}")
    
    # Build source objects (Chunks)
    sources = [
        {
            "id": r.VectorEntry.id, 
            "doc_id": r.VectorEntry.doc_id, 
            "content": r.VectorEntry.content,
            "score": round(1 - r.dist, 3), 
            "filename": r.VectorEntry.document.filename if r.VectorEntry.document else "Unknown",
        }
        for r in valid_results
    ]

    # 3. Hybrid Logic Decision
    has_context = len(valid_results) > 0
    context_chunks = [r["content"] for r in sources] if has_context else []
    
    try:
        is_conv = not has_context  # pure conversational if no docs found
        answer = await asyncio.to_thread(ai_service.generate_answer, req.input, context_chunks, db, history_data_list, is_conv)

        # Strict Output Parsing: Validate against engine failure or safety fallback
        expected_fallback = ProviderFactory._config_cache.get("fallback_message", "")
        if answer.startswith("Error:") or answer.startswith("I am currently experiencing a processing error."):
            mode = "error"
        elif expected_fallback and answer.strip() == expected_fallback.strip():
            mode = "fallback"
        else:
            mode = "semantic" if has_context else "conversational"
            
    except Exception as e:
        logger.error(f"AI Generation Error: {str(e)}")
        answer = ProviderFactory._config_cache.get("fallback_message", "I am currently experiencing a processing error. Please retry your query shortly.")
        mode = "error"

    # Parallel Post-Processing: Title + Follow-ups
    background_tasks = []
    
    # Task: Title generation for first message
    if not history_data:
        background_tasks.append(asyncio.to_thread(ai_service.generate_chat_title, req.input, db))
    else:
        background_tasks.append(asyncio.sleep(0, result=None)) # placeholder

    # Task: Dynamic Follow-ups Generation
    if mode in ["semantic", "conversational", "fallback"]:
        background_tasks.append(asyncio.to_thread(ai_service.generate_followups, answer, req.input, context_chunks, db, history_data))
    else:
        background_tasks.append(asyncio.sleep(0, result=[])) # placeholder

    # Await all background tasks concurrently
    results_async = await asyncio.gather(*background_tasks, return_exceptions=True)
    
    # Process Results
    new_title = results_async[0] if not isinstance(results_async[0], Exception) else None
    follow_ups = results_async[1] if not isinstance(results_async[1], Exception) else []

    if new_title:
        session.title = new_title

    # Save History — stores context chunks as 'sources' like before
    new_history = ChatHistory(
        session_id=session_id,
        query=req.input,
        answer=answer,
        context_used=sources
    )
    db.add(new_history)
    session.updated_at = datetime.utcnow()
    db.commit()

    return {
        "answer": answer,
        "session_id": session_id,
        "session_title": session.title,
        "sources": sources,
        "mode": mode,
        "follow_ups": follow_ups,
        "latency": round(time.time() - start_time, 2)
    }

# --- Sessions & History ---
@app.get("/sessions", tags=["Admin"])
def list_sessions(
    internal_only: bool = False,
    scope: Optional[str] = "all",  # all | admin | public
    sort: Optional[str] = "newest",  # newest | oldest
    search: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    user: str = Depends(get_current_user)
):
    query = db.query(ChatSession)
    if search:
        query = query.filter(
            (ChatSession.title.ilike(f"%{search}%")) | (ChatSession.id.ilike(f"%{search}%"))
        )
    if internal_only or scope == "admin":
        query = query.filter(ChatSession.is_internal == True)
    elif scope == "public":
        query = query.filter(ChatSession.is_internal == False)
    
    if sort == "oldest":
        query = query.order_by(ChatSession.updated_at)
    else:
        query = query.order_by(desc(ChatSession.updated_at))
    
    total = query.count()
    results = query.offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size, "items": results}

@app.get("/sessions/{session_id}", tags=["Admin"])
def get_session_history(session_id: str, db: Session = Depends(get_db), user: str = Depends(get_current_user)):
    return db.query(ChatHistory).filter(ChatHistory.session_id == session_id).order_by(ChatHistory.timestamp).all()

@app.get("/public/sessions/{session_id}/history", tags=["Public"])
def get_public_session_history(session_id: str, db: Session = Depends(get_db)):
    """Publicly accessible history for guest sessions only."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        return []
    
    # Security: If it's an internal session, it cannot be accessed without auth
    if session.is_internal:
        raise HTTPException(status_code=403, detail="Unauthorized access for internal session")
        
    return db.query(ChatHistory).filter(ChatHistory.session_id == session_id).order_by(ChatHistory.timestamp).all()

# --- Admin & Management ---

@app.get("/admin/dashboard", tags=["Admin"])
def admin_dashboard_stats(db: Session = Depends(get_db), user: str = Depends(get_current_user)):
    doc_count = db.query(Document).filter(Document.is_deleted == False).count()
    chunk_count = db.query(VectorEntry).join(Document).filter(Document.is_deleted == False).count()
    chat_count = db.query(ChatHistory).count()
    return {
        "total_documents": doc_count,
        "total_vector_chunks": chunk_count,
        "total_chats_handled": chat_count
    }

@app.get("/admin/documents", tags=["Admin"])
def list_documents(
    category: Optional[str] = None, 
    search: Optional[str] = None, 
    page: int = 1,
    page_size: int = 15,
    db: Session = Depends(get_db),
    user: str = Depends(get_current_user)
):
    query = db.query(Document).filter(Document.is_deleted == False)
    if category: query = query.filter(Document.category == category)
    if search: query = query.filter(Document.filename.ilike(f"%{search}%"))
    
    total = query.count()
    items = query.order_by(desc(Document.created_at)).offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size, "items": items}

@app.get("/admin/trash", tags=["Admin"])
def list_trash(
    search: Optional[str] = None,
    source_type: Optional[str] = None,
    page: int = 1,
    page_size: int = 15,
    db: Session = Depends(get_db),
    user: str = Depends(get_current_user)
):
    query = db.query(Document).filter(Document.is_deleted == True)
    if search: query = query.filter(Document.filename.ilike(f"%{search}%"))
    if source_type: query = query.filter(Document.source_type == source_type)
    
    total = query.count()
    items = query.order_by(desc(Document.deleted_at)).offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size, "items": items}

@app.post("/admin/restore/{doc_id}", tags=["Admin"])
def restore_doc(doc_id: int, db: Session = Depends(get_db), user: str = Depends(get_current_user)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc: raise HTTPException(status_code=404)
    doc.is_deleted = False
    doc.deleted_at = None
    db.commit()
    return {"status": "success"}

@app.delete("/documents/{doc_id}", tags=["Admin"])
def soft_delete_doc(doc_id: int, db: Session = Depends(get_db), user: str = Depends(get_current_user)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc: raise HTTPException(status_code=404)
    doc.is_deleted = True
    doc.deleted_at = datetime.utcnow()
    db.commit()
    return {"status": "success"}

@app.delete("/admin/permanent/{doc_id}", tags=["Admin"])
def permanent_delete(doc_id: int, db: Session = Depends(get_db), user: str = Depends(get_current_user)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc: raise HTTPException(status_code=404)
    db.delete(doc) # Cascade handles VectorEntry
    db.commit()
    return {"status": "success"}

@app.get("/admin/config", tags=["Admin"])
def get_config(db: Session = Depends(get_db), user: str = Depends(get_current_user)):
    return {s.key: s.value for s in db.query(AppSettings).all()}

@app.post("/admin/config", tags=["Admin"])
def update_config(req: ConfigUpdate, db: Session = Depends(get_db), user: str = Depends(get_current_user)):
    updates = req.dict(exclude_none=True)
    for k, v in updates.items():
        setting = db.query(AppSettings).filter(AppSettings.key == k).first()
        if setting: setting.value = str(v)
        else: db.add(AppSettings(key=k, value=str(v)))
    db.commit()
    
    # Critical: Reload Config Cache in Factory immediately after update
    ProviderFactory.reload_config(db)
    
    return {"status": "success"}

# --- Auth Endpoints ---
@app.post("/auth/login", tags=["Auth"])
def login(req: LoginRequest, response: __import__('fastapi').Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username).first()
    
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    valid_user = user.username
    from auth import ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS
    access_token = create_access_token(data={"sub": valid_user})
    refresh_token = create_refresh_token(data={"sub": valid_user})
    
    response.set_cookie(
        key="oracle_refresh",
        value=refresh_token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=60 * 60 * 24 * REFRESH_TOKEN_EXPIRE_DAYS
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": valid_user,
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@app.post("/auth/change-password", tags=["Auth"])
def change_password(req: PasswordChangeRequest, db: Session = Depends(get_db), username: str = Depends(get_current_user)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if not verify_password(req.current_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect current password")
    
    user.hashed_password = get_password_hash(req.new_password)
    db.commit()
    
    logger.info(f"Password changed successfully for user: {username}")
    return {"status": "success", "message": "Password updated successfully."}

@app.post("/auth/refresh", tags=["Auth"])
def refresh_token_endpoint(request: Request, response: __import__('fastapi').Response):
    """Exchange a valid refresh token (httpOnly cookie) for a new access token."""
    refresh_cookie = request.cookies.get("oracle_refresh")
    if not refresh_cookie:
        raise HTTPException(status_code=401, detail="No refresh token")
    payload = decode_refresh_token(refresh_cookie)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    username = payload.get("sub")
    access_token = create_access_token(data={"sub": username})
    from auth import ACCESS_TOKEN_EXPIRE_MINUTES
    return {"access_token": access_token, "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60}

@app.post("/auth/logout", tags=["Auth"])
def logout(response: __import__('fastapi').Response):
    response.delete_cookie("oracle_refresh")
    return {"status": "logged out"}

@app.get("/auth/me", tags=["Auth"])
def get_me(username: str = Depends(get_current_user)):
    import os
    return {"username": username, "display_name": os.getenv("ADMIN_DISPLAY_NAME", username)}

# --- Sessions Delete ---
@app.delete("/sessions/{session_id}", tags=["Admin"])
def delete_session(session_id: str, db: Session = Depends(get_db), user: str = Depends(get_current_user)):
    sess = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not sess: raise HTTPException(status_code=404)
    db.delete(sess)
    db.commit()
    return {"status": "success"}

@app.post("/sessions/bulk-delete", tags=["Admin"])
def bulk_delete_sessions(req: BulkDeleteRequest, db: Session = Depends(get_db), user: str = Depends(get_current_user)):
    db.query(ChatSession).filter(ChatSession.id.in_(req.ids)).delete(synchronize_session=False)
    db.commit()
    return {"status": "success", "deleted": len(req.ids)}

# --- Bulk Document Delete ---
@app.post("/documents/bulk-delete", tags=["Admin"])
def bulk_delete_documents(req: BulkDeleteRequest, db: Session = Depends(get_db), user: str = Depends(get_current_user)):
    now = datetime.utcnow()
    db.query(Document).filter(Document.id.in_(req.ids)).update(
        {"is_deleted": True, "deleted_at": now}, synchronize_session=False
    )
    db.commit()
    return {"status": "success", "deleted": len(req.ids)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
