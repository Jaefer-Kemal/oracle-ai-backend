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
import json
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

class AIService:
    def __init__(self):
        pass

    def _get_model(self) -> str:
        requested_model = ProviderFactory._config_cache.get("grok_model", "grok-3-auto")
        
        # Validation: Fallback if model is not in the allowed list
        if requested_model not in _Models.models.keys():
            return "grok-3-auto"
            
        return requested_model

    def decontextualize_query(self, query: str, history: List[Dict[str, str]], db: Session) -> str:
        """Industry Standard (Rewrite-Retrieve-Read): Extract core search intent from conversational input."""
        try:
            # Short-circuit for vague follow-ups if history exists
            vague_terms = ["tell me more", "give me more", "more information", "what else", "continue", "elaborate"]
            if query.lower().strip() in vague_terms and history:
                # Use the last query as the baseline for retrieval
                return history[-1]['q']

            history_str = "\n".join([f"User: {h['q']}\nAssistant: {h['a']}" for h in history[-5:]]) if history else "None"
            
            prompt = f"""You are a query extraction specialist. Your task is to extract a clean, standalone search query from the user's message.

<rules>
1. REMOVE all personal information (e.g., "my name is...", "I am...", "call me...")
2. REMOVE all conversational openers (e.g., "I want to know", "can you tell me", "please explain")
3. EXTRACT only the core topic, entity, or question.
4. RESOLVE pronouns using the conversation history (e.g., "tell me more about it" -> use the topic from history).
5. If the user asks for "more" or "continue", identify exactly what topic they are referring to from the history.
6. Return ONLY the clean search query. No quotes, no explanations, no conversational filler.
</rules>

<history>
{history_str}
</history>

<user_message>
{query}
</user_message>

Clean Search Query:"""
            
            # Use FLASH-LITE for decontextualization (2026 Stable ID)
            response = ProviderFactory.generate_answer(db, prompt, history, model_override="gemini-flash-lite-latest")
            result = response.get("response", query).strip().replace('"', '').replace("'", "").split('\n')[0].strip()
            # Sanity check: don't return an empty or obviously broken query
            return result if len(result) > 3 else query
        except:
            return query

    def generate_answer(self, query: str, context: List[str], db: Session, history: List[Dict[str, str]] = None, is_conversational: bool = False) -> str:
        """Generation with Chat History and specific RAG Prompt."""
        from models import AppSettings
        try:
            # Format Context
            context_str = "\n\n".join([f"--- Source {i+1} ---\n{c}" for i, c in enumerate(context)])
            
            # Format History
            history_str = ""
            if history:
                history_text = "\n".join([f"User: {h['q']}\nAssistant: {h['a']}" for h in history])
                history_str = f"<history>\n{history_text}\n</history>\n"

            # Fetch fallback from Factory Cache
            fallback_text = ProviderFactory._config_cache.get("fallback_message", "I'm sorry, I don't have information on that.")

            prompt = f"""<system_directive>
You are the internal AI Knowledge Assistant for Oracle AI Solutions.
 
Your mission is to provide highly accurate information based EXCLUSIVELY on the provided <retrieved_context> for factual questions. You must maintain a professional yet comfortably casual and approachable tone—think of yourself as a knowledgeable colleague who is both helpful and expert.
 
**Special Handling:** 
- **Greetings/Small Talk**: You are allowed to respond naturally to greetings or casual remarks (e.g., "Hi", "How's it going?") without requiring context.
- **Vague Queries ("Tell me more", "Give me more")**: If a user asks for more information and context exists, provide deeper technical details or related trivia from the context. If no clear context exists but there is conversation history, clarify the topic: "I'd be happy to share more! Are we still discussing [Topic from History], or would you like to explore something new?"
</system_directive>

<tone_and_style_guidelines>
1. Be Conversational: Speak naturally and casually. Use friendly greetings (e.g., "Hey there!", "Happy to help with that!", "Sure thing!").
2. No Robotic/Technical Language: Never mention "context", "knowledge base", "retrieved documents", or "databases". Never say "No relevant context found".
3. Be Professional: Even when casual, ensure your grammar is perfect and your information is delivered clearly.
4. Be Concise: Answer exactly what is asked. Use bullet points or lists for multi-part information.
</tone_and_style_guidelines>

<strict_grounding_rules>
1. THE CONTEXT IS YOUR ONLY TRUTH: For factual queries, you must base your answers entirely on the <retrieved_context>.
2. NO HALLUCINATION: If a factual detail is not in the context, you must not invent it.
3. NO EXTERNAL SCOPE: Refuse to discuss external politics, creative fiction, or generate non-company code.
</strict_grounding_rules>

<fallback_protocol>
If the <retrieved_context> does not contain the specific answer to a factual question, you must handle it gracefully with a professional, dynamic lead-in. Avoid robotic "I don't know" responses.
 
- **Partial Match**: "While our primary records on [Strict Topic] are currently being updated, here is what I can share regarding the related areas I found..."
- **Complete Miss**: Naturally acknowledge the topic and explain the limitation. "I've carefully reviewed our internal knowledge base for details on [Topic], but it seems we don't have those specific records on file at the moment. {fallback_text}"
- **Vague Follow-up**: If context is provided but the user just says "tell me more," dive into the secondary details of the provided context (prices, locations, technical specs) that weren't mentioned in the previous turn.
</fallback_protocol>

<examples>
  <example_1>
    <scenario>Simple Greeting</scenario>
    <user_query>Hi there!</user_query>
    <retrieved_context>[EMPTY]</retrieved_context>
    <ideal_response>Hey! I'm here to help you navigate our company knowledge base. What can I look up for you today?</ideal_response>
  </example_1>

  <example_2>
    <scenario>Vague Query (No Context)</scenario>
    <user_query>Tell me more.</user_query>
    <retrieved_context>[EMPTY]</retrieved_context>
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
    <retrieved_context>[EMPTY]</retrieved_context>
    <ideal_response>I've checked our internal records, but I couldn't find any specific details on the Q3 Marketing Campaign launch yet. {fallback_text}</ideal_response>
  </example_4>
</examples>

<input_data>
<conversation_history>
{history_str if history_str else "No conversation history."}
</conversation_history>

<retrieved_context>
{context_str if context_str else "[EMPTY]"}
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
                return ProviderFactory._config_cache.get("fallback_message", "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again in a moment.")
            
            final_text = response.get("response")
            return final_text or "Error: No response from generation engine."
        except Exception as e:
            logger.error(f"AI Generate Answer Error: {e}")
            # Dynamic Fallback from Cache
            return ProviderFactory._config_cache.get("fallback_message", "I am currently experiencing a processing error. Please retry your query shortly.")

    def generate_chat_title(self, first_query: str, db: Session) -> str:
        """Generate a short (3-5 word) summary for the chat session title."""
        try:
            prompt = f"Summarize the following user query into a 3 to 5 word professional chat title. Only provide the title text, nothing else.\n\nQuery: {first_query}\n\nTitle:"
            # Use FLASH-LITE for title generation
            response = ProviderFactory.generate_answer(db, prompt, model_override="gemini-flash-lite-latest")
            title = response.get("response", "New Conversation").strip().replace('"', '')
            return title[:50] # Safety limit
        except Exception as e:
            logger.error(f"Grok Generate Title Error: {e}")
            return "Professional Session"

    def generate_followups(self, answer: str, query: str, context: List[str], db: Session, history: List[dict] = None) -> List[str]:
        """Generate exactly 3 extremely short follow-up questions using robust Regex parsing."""
        import re
        fallback_suggestions = []
        try:
            suggested_raw = ProviderFactory._config_cache.get("suggested_questions", "[]")
            if isinstance(suggested_raw, (str, list)):
                if isinstance(suggested_raw, str):
                    try:
                        fallback_suggestions = json.loads(suggested_raw)
                    except:
                        fallback_suggestions = []
                else:
                    fallback_suggestions = suggested_raw
        except:
            pass

        try:
            # Check for triggers to avoid "hallucinated" or "off-topic" AI suggestions
            greetings = ["hi", "hello", "hey", "hola", "greetings", "hi there", "good morning", "good afternoon"]
            is_greeting = query.lower().strip().rstrip("?!.") in greetings
            
            # Refusal keywords (AI safety or inability to answer)
            refusal_keywords = ["cannot answer", "unavailable", "illegal", "not allowed", "sorry", "no relevant context"]
            is_refusal = any(k in answer.lower() for k in refusal_keywords)

            # RULE: If there is NO RAG context, or if the AI is refusing/greeting, 
            # do not ask the AI for creative follow-ups. Revert to the high-quality fallbacks.
            if (not context or is_refusal or is_greeting) and fallback_suggestions:
                return fallback_suggestions[:3]

            # Critical check for valid answer length to avoid parsing empty/error strings
            if not answer or len(answer) < 20: 
                return fallback_suggestions[:3]

            context_str = "\n".join(context)[:2000]
            prompt = f"""You are a copywriter for user intent.
Your goal is to provide exactly 3 extremely brief follow-up questions or requests that a HUMAN USER would say next.

<rules>
1. USE FIRST PERSON: Always phrase from the user's perspective (e.g., "Tell me about...", "I want to see...").
2. TOPIC DIVERSITY: Each of the 3 suggestions must be independent and cover different angles of the context (e.g., don't suggest 3 versions of "How much does it cost?").
3. DO NOT use "You": Phrases like "You can ask about" are forbidden.
4. BE PROACTIVE: Suggestions should be new inquiries for further exploration.
5. Output EXACTLY 3 items, one per line.
6. NO numbering.
</rules>

<user_query>{query}</user_query>
<context>{context_str}</context>
<answer>{answer}</answer>

Output (exactly 3 questions, 1 per line):"""
            # Use FLASH-LITE for follow-ups
            response = ProviderFactory.generate_answer(db, prompt, model_override="gemini-flash-lite-latest")
            result = response.get("response", "").strip()
            
            # Robust Parsing Strategy:
            # We now strictly split by NEWLINES to avoid fragments from commas or internal question marks.
            raw_parts = re.split(r'\n+', result)
            
            cleaned = []
            for p in raw_parts:
                # Clean up: remove numbering, quotes, and whitespace
                p_clean = re.sub(r'^\d+[\.\)]\s*', '', p.strip()) # Remove "1. " or "1) "
                p_clean = p_clean.replace('"', '').replace("'", "").strip()
                
                # Add back the question mark if it was lost in split
                if p_clean and not p_clean.endswith('?'):
                    p_clean += '?'
                
                if p_clean and len(p_clean) > 5:
                    cleaned.append(p_clean)

            # Ensure exactly 3
            if len(cleaned) < 3:
                # Add fallbacks if AI under-generated
                for fs in fallback_suggestions:
                    if fs not in cleaned:
                        cleaned.append(fs)
                    if len(cleaned) >= 3: break
            
            return cleaned[:3]
        except Exception as e:
            logger.error(f"Generate Followups Error: {e}")
            return fallback_suggestions[:3]


    def suggest_category(self, content: str, db: Session) -> str:
        """Analyze document content and suggest a 1-2 word professional category."""
        try:
            # Use small sample of content
            sample = content[:4000]
            prompt = f"Analyze this text and suggest a single professional category (1-2 words). Examples: HR Policy, Technical Manual, Sales Strategy, Legal, Operations. ONLY provide the category name.\n\nText: {sample}\n\nCategory:"
            # Use FLASH-LITE for categorization
            response = ProviderFactory.generate_answer(db, prompt, model_override="gemini-flash-lite-latest")
            category = response.get("response", "General").strip().replace('"', '').replace(".", "")
            return category if len(category) > 1 else "General"
        except Exception as e:
            logger.error(f"Suggest Category Error: {e}")
            return "General"

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
