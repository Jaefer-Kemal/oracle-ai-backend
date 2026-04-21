import os
import time
import hashlib
from typing import List, Dict, Optional
import cohere
from pypdf import PdfReader
from docx import Document as DocxDocument
from dotenv import load_dotenv

from core.providers.factory import ProviderFactory
from core.providers.grok import Grok, _Models
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

            prompt = f"""<system_directive>
You are the internal AI Knowledge Assistant for the Company.

Your mission is to provide highly accurate information based EXCLUSIVELY on the provided <retrieved_context> for factual questions. You must maintain a professional yet comfortably casual and approachable tone—think of yourself as a knowledgeable colleague who is both helpful and expert.

**Special Handling:** 
- **Greetings/Small Talk**: You are allowed to respond naturally to greetings or casual remarks (e.g., "Hi", "How's it going?") without requiring context.
- **Vague Queries**: If a user asks something vague (e.g., "tell me more", "what else?") and there is no clear context or history to link it to, politely ask for clarification (e.g., "I'd love to help! Could you specify what you'd like to know more about regarding company policies?") instead of triggering a fallback.
</system_directive>

<tone_and_style_guidelines>
1. Be Conversational: Speak naturally and casually. Use friendly greetings (e.g., "Hey there!", "Happy to help with that!", "Sure thing!").
2. No Robotic Fallbacks: Never start a response with just a fallback message. Always acknowledge the user's intent first.
3. Be Professional: Even when casual, ensure your grammar is perfect and your information is delivered clearly.
4. Be Concise: Answer exactly what is asked. Use bullet points or lists for multi-part information.
</tone_and_style_guidelines>

<strict_grounding_rules>
1. THE CONTEXT IS YOUR ONLY TRUTH: For factual queries, you must base your answers entirely on the <retrieved_context>.
2. NO HALLUCINATION: If a factual detail is not in the context, you must not invent it.
3. NO EXTERNAL SCOPE: Refuse to discuss external politics, creative fiction, or generate non-company code.
</strict_grounding_rules>

<fallback_protocol>
If the <retrieved_context> does not contain the specific answer to a factual question, you must handle it gracefully with a professional, dynamic lead-in.

- Partial Match: Share what you *do* find in the records, and then pivot naturally to explain what is missing.
- Complete Miss: Naturally apologize. Phrase this dynamically based on the query (e.g., "I've looked through our current records on [Topic], but I couldn't find those specific details."). You MUST APPEND the exact fallback message at the very end.
- System Fallback to Append: "{fallback_text}"
</fallback_protocol>

<examples>
  <example_1>
    <scenario>Simple Greeting</scenario>
    <user_query>Hi there!</user_query>
    <retrieved_context>No relevant context found.</retrieved_context>
    <ideal_response>Hey! I'm here to help you navigate our company knowledge base. What can I look up for you today?</ideal_response>
  </example_1>

  <example_2>
    <scenario>Vague Query (No Context)</scenario>
    <user_query>Tell me more.</user_query>
    <retrieved_context>No relevant context found.</retrieved_context>
    <ideal_response>I'd be happy to tell you more! Could you let me know which specific topic or policy you're interested in? That way I can give you the most accurate details.</ideal_response>
  </example_2>

  <example_3>
    <scenario>Perfect Context Match</scenario>
    <user_query>What's the policy for business travel?</user_query>
    <retrieved_context>All business travel must be booked through the travel portal. Flights over 5 hours are eligible for business class upgrades.</retrieved_context>
    <ideal_response>Sure thing! For business travel, the policy is that everything needs to be booked through our travel portal. Also, if your flight is over 5 hours, you're eligible for a business class upgrade. Let me know if you need help with anything else!</ideal_response>
  </example_3>

  <example_4>
    <scenario>Complete Fallback</scenario>
    <user_query>What are the details of the Q3 Marketing Campaign launch?</user_query>
    <retrieved_context>No relevant context found.</retrieved_context>
    <ideal_response>I've checked our internal records, but I couldn't find any specific details on the Q3 Marketing Campaign launch yet. {fallback_text}</ideal_response>
  </example_4>
</examples>

<input_data>
<conversation_history>
{history_str if history_str else "No conversation history."}
</conversation_history>

<retrieved_context>
{context_str if context_str else "No relevant context found."}
</retrieved_context>
</input_data>

<execution>
Analyze the user query and the <retrieved_context>. Deliver your response strictly following the <tone_and_style_guidelines> and <fallback_protocol>.

User Query: {query}
</execution>"""
        
            # Modular Provider Generation with Fallback
            response = ProviderFactory.generate_answer(db, prompt, history)
            
            # DEBUG: Log response keys to identify structure issues
            logger.info(f"AI Response Keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
            
            if "error" in response:
                logger.error(f"Provider Failure (All fallbacks exhausted): {response['error']}")
                fb_msg = db.query(AppSettings).filter(AppSettings.key == "fallback_message").first()
                return fb_msg.value if fb_msg else "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again in a moment."
            
            final_text = response.get("response")
            return final_text or "Error: No response from generation engine."
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

    def generate_followups(self, answer: str, query: str, context: List[str]) -> List[str]:
        """Generate 3 extremely short follow-up questions based on the query, context, and answer using minimal latency grok-3-fast."""
        try:
            context_str = "\n".join(context)[:2000] # Cap context so prompt isn't too huge
            grok_client = Grok("grok-3-fast")
            prompt = f"""You are an expert UX researcher analyzing how users explore information.
Based on the user's original query, the provided context, and the AI's answer, generate exactly 3 highly specific, engaging follow-up questions a user would logically ask next to dive deeper.

<rules>
1. Questions MUST strictly reference specific entities, nouns, or concepts mentioned in the context or answer.
2. AVOID generic questions (e.g., "Tell me more", "How does it work?", "What are the rules?").
3. Focus on practical application, limitations, or deeper exploration of the topic that the user is currently asking about.
4. Keep questions concise (Maximum 8 words per question).
5. Output ONLY a comma-separated list of the 3 questions. No numbering, no introduction.
</rules>

<user_query>
{query}
</user_query>

<context>
{context_str}
</context>

<ai_answer>
{answer}
</ai_answer>

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
