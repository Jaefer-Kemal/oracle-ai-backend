import os
import time
import hashlib
from typing import List, Dict, Optional
import cohere
from pypdf import PdfReader
from docx import Document as DocxDocument
from dotenv import load_dotenv

from core import Grok
import logging
from sqlalchemy.orm import Session

logger = logging.getLogger("rag-backend.services")

load_dotenv()

class CohereService:
    def __init__(self):
        # SDK v5+: use ClientV2
        self.client = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
        # Rate limit: 100 calls/min = ~1.67/sec. We target 90/min to be safe = ~0.67s/call
        self.rate_limit_pause = 0.7  # seconds between batches
        self.max_batch_size = 30     # smaller batches = better rate control at 100/min

    def get_embeddings(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            print(f"   [Cohere] Embedding batch {i//self.max_batch_size + 1}... ({len(batch)} chunks) [Type: {input_type}]")
            
            max_retries = 5
            base_delay = 2.0
            for attempt in range(max_retries):
                try:
                    # SDK v5 ClientV2: requires embedding_types param, returns response.embeddings.float
                    response = self.client.embed(
                        texts=batch,
                        model="embed-v4.0",
                        input_type=input_type,
                        embedding_types=["float"]
                    )
                    all_embeddings.extend(response.embeddings.float)
                    break
                except Exception as e:
                    if "too many requests" in str(e).lower() and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"   [Cohere] Rate limit hit. Retrying in {delay}s... (Attempt {attempt + 1})")
                        time.sleep(delay)
                    else:
                        raise e
            
            if i + self.max_batch_size < len(texts):
                time.sleep(self.rate_limit_pause)
        return all_embeddings

class GrokService:
    def __init__(self):
        pass

    def _get_model(self, db: Session) -> str:
        from models import AppSettings
        from core.grok import _Models
        
        setting = db.query(AppSettings).filter(AppSettings.key == "grok_model").first()
        requested_model = setting.value if setting else "grok-3-auto"
        
        # Validation: Fallback if model is not in the allowed list
        if requested_model not in _Models.models.keys():
            return "grok-3-auto"
            
        return requested_model

    def decontextualize_query(self, query: str, history: List[Dict[str, str]], db: Session) -> str:
        """Industry Standard (Rewrite-Retrieve-Read): Extract core search intent from conversational input."""
        try:
            model = self._get_model(db)
            grok_client = Grok(model)
            history_str = "\n".join([f"User: {h['q']}\nAssistant: {h['a']}" for h in history[-5:]]) if history else "None"
            
            prompt = f"""You are a query extraction specialist. Your task is to extract a clean, standalone search query from the user's message.

<rules>
1. REMOVE all personal information (e.g., "my name is...", "I am...", "call me...")
2. REMOVE all conversational openers (e.g., "I want to know", "can you tell me", "please explain")
3. EXTRACT only the core topic, entity, or question.
4. RESOLVE pronouns using the conversation history (e.g., "tell me more about it" -> use the topic from history).
5. If the message is a short follow-up (e.g., "are you sure?", "explain that"), extract the LAST TOPIC from the history.
6. Return ONLY the clean search query. No quotes, no explanations, no conversational filler.
</rules>

<history>
{history_str}
</history>

<user_message>
{query}
</user_message>

Clean Search Query:"""
            
            response = grok_client.start_convo(prompt)
            result = response.get("response", query).strip().replace('"', '').replace("'", "").split('\n')[0].strip()
            # Sanity check: don't return an empty or obviously broken query
            return result if len(result) > 3 else query
        except:
            return query

    def generate_answer(self, query: str, context: List[str], db: Session, history: List[Dict[str, str]] = None, is_conversational: bool = False) -> str:
        """Generation with Chat History and specific RAG Prompt."""
        from models import AppSettings
        try:
            model = self._get_model(db)
            grok_client = Grok(model)
            
            # Format Context
            context_str = "\n\n".join([f"--- Source {i+1} ---\n{c}" for i, c in enumerate(context)])
            
            # Format History
            history_str = ""
            if history:
                history_text = "\n".join([f"User: {h['q']}\nAssistant: {h['a']}" for h in history])
                history_str = f"<history>\n{history_text}\n</history>\n"

            # Fetch fallback from DB
            fb_msg = db.query(AppSettings).filter(AppSettings.key == "fallback_message").first()
            fallback_text = fb_msg.value if fb_msg else "I'm sorry, I don't have information on that."

            # Unified Constraint: Allow Grok to use both Context and History intelligently
            c1 = f"Answer the <query> using the <context> and <history>. For factual questions, synthesize a professional summary strictly from the <context>. For personal or conversational questions (e.g., 'what is my name'), you MUST use the <history>. If NEITHER contains the answer, output ONLY the exact fallback message: '{fallback_text}'"

            prompt = f"""You are a professional corporate AI assistant named Oracle.

<instructions>
1. {c1}
2. Do not use external information, assumptions, or pre-trained knowledge to answer the question.
3. If the answer is not present in the provided information, use the fallback message and output NOTHING else.
4. Maintain a professional, concise, and helpful tone (ideally 100-200 words).
</instructions>

{history_str}
<context>
{context_str}
</context>

<query>
{query}
</query>

Assistant Answer:"""
        
            response = grok_client.start_convo(prompt)
            return response.get("response", "Error: No response from generation engine.")
        except Exception as e:
            logger.error(f"Grok Generate Answer Error: {e}")
            # Dynamic Fallback from DB
            fb_msg = db.query(AppSettings).filter(AppSettings.key == "fallback_message").first()
            return fb_msg.value if fb_msg else "I am currently experiencing a processing error. Please retry your query shortly."

    def generate_chat_title(self, first_query: str, db: Session) -> str:
        """Generate a short (3-5 word) summary for the chat session title."""
        try:
            model = self._get_model(db)
            grok_client = Grok(model)
            prompt = f"Summarize the following user query into a 3 to 5 word professional chat title. Only provide the title text, nothing else.\n\nQuery: {first_query}\n\nTitle:"
            response = grok_client.start_convo(prompt)
            title = response.get("response", "New Conversation").strip().replace('"', '')
            return title[:50] # Safety limit
        except Exception as e:
            logger.error(f"Grok Generate Title Error: {e}")
            return "Professional Session"

    def generate_followups(self, answer: str) -> List[str]:
        """Generate 3 extremely short follow-up questions based on the answer using minimal latency grok-3-fast."""
        try:
            # Hardcode fast model for UX responsiveness
            grok_client = Grok("grok-3-fast")
            prompt = f"""You are an expert UX researcher analyzing how users explore information.
Based on the following AI answer, generate exactly 3 highly specific, engaging follow-up questions a user would logically ask next to dive deeper.

<rules>
1. Questions MUST strictly reference specific entities, nouns, or concepts mentioned in the text.
2. AVOID generic questions (e.g., "Tell me more", "How does it work?", "What are the rules?").
3. Focus on practical application, limitations, or deeper exploration of the topic.
4. Keep questions concise (Maximum 8 words per question).
5. Output ONLY a comma-separated list of the 3 questions. No numbering, no introduction.
</rules>

<example_bad>
What is it?, Tell me more, How does it work?
</example_bad>

<example_good>
How are vector embeddings securely updated?, Can I upload scanned PDF files?, What is the maximum timeout limit?
</example_good>

<answer>
{answer}
</answer>

Follow-up questions:"""
            response = grok_client.start_convo(prompt)
            result = response.get("response", "").strip()
            
            # Parse CSV to list
            questions = [q.strip() for q in result.split(",") if q.strip()]
            return questions[:3]
        except Exception as e:
            logger.error(f"Grok Followups Error: {e}")
            return []

class ParserService:
    @staticmethod
    def extract_text_from_pdf(content: bytes) -> str:
        from io import BytesIO
        reader = PdfReader(BytesIO(content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    @staticmethod
    def extract_text_from_docx(content: bytes) -> str:
        from io import BytesIO
        doc = DocxDocument(BytesIO(content))
        return "\n".join([para.text for para in doc.paragraphs])

    @staticmethod
    def get_file_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200) -> List[str]:
        """Larger chunks for embed-v4.0 (128k token context window)."""
        chunks = []
        i = 0
        while i < len(text):
            chunks.append(text[i:i + chunk_size])
            i += (chunk_size - overlap)
        return [c.strip() for c in chunks if len(c.strip()) > 20]
