from core        import Log, Run, Utils, Parser, Signature, Anon, Headers
from curl_cffi   import requests, CurlMime
from dataclasses import dataclass, field
from bs4         import BeautifulSoup
from json        import dumps, loads
from secrets     import token_hex
from uuid        import uuid4

@dataclass
class Models:
    models: dict[str, list[str]] = field(default_factory=lambda: {
        "grok-3-auto": ["MODEL_MODE_AUTO", "auto"],
        "grok-3-fast": ["MODEL_MODE_FAST", "fast"],
        "grok-4": ["MODEL_MODE_EXPERT", "expert"],
        "grok-4-mini-thinking-tahoe": ["MODEL_MODE_GROK_4_MINI_THINKING", "grok-4-mini-thinking"]
    })

    def get_model_mode(self, model: str, index: int) -> str:
        return self.models.get(model, ["MODEL_MODE_AUTO", "auto"])[index]

_Models = Models()

class Grok:
    
    
    def __init__(self, model: str = "grok-3-auto", proxy: str = None) -> None:
        self.session: requests.session.Session = requests.Session(impersonate="chrome120", default_headers=False)
        self.headers: Headers = Headers()
        
        # Validation: Ensure the model exists in the supported list, otherwise fallback to auto
        if model not in _Models.models:
            model = "grok-3-auto"
            
        self.model_mode: str = _Models.get_model_mode(model, 0)
        self.model: str = model
        self.mode: str = _Models.get_model_mode(model, 1)
        self.c_run: int = 0
        self.keys: dict = Anon.generate_keys()
        if proxy:
            self.session.proxies = {
                "all": proxy
            }
    
    def _load(self, extra_data: dict = None) -> None:
        
        if not extra_data:
            self.session.headers = self.headers.LOAD
            load_site: requests.models.Response = self.session.get('https://grok.com/c')
            self.session.cookies.update(load_site.cookies)
            
            scripts: list = [s['src'] for s in BeautifulSoup(load_site.text, 'html.parser').find_all('script', src=True) if '/_next/static/chunks/' in s['src']]

            self.actions, self.xsid_script = Parser.parse_grok(scripts)
            
            self.baggage: str = Utils.between(load_site.text, '<meta name="baggage" content="', '"')
            self.sentry_trace: str = Utils.between(load_site.text, '<meta name="sentry-trace" content="', '-')
        else:
            self.session.cookies.update(extra_data["cookies"])

            self.actions: list = extra_data["actions"]
            self.xsid_script: list =  extra_data["xsid_script"]
            self.baggage: str = extra_data["baggage"]
            self.sentry_trace: str = extra_data["sentry_trace"]
            
    
    def c_request(self, next_action: str) -> None:
        
        self.session.headers = self.headers.C_REQUEST
        self.session.headers.update({
            'baggage': self.baggage,
            'next-action': next_action,
            'sentry-trace': f'{self.sentry_trace}-{str(uuid4()).replace("-", "")[:16]}-0',
        })
        self.session.headers = Headers.fix_order(self.session.headers, self.headers.C_REQUEST)
        
        if self.c_run == 0:
            self.session.headers.pop("content-type")
            
            mime = CurlMime()
            mime.addpart(name="1", data=bytes(self.keys["userPublicKey"]), filename="blob", content_type="application/octet-stream")
            mime.addpart(name="0", filename=None, data='[{"userPublicKey":"$o1"}]')
            
            c_request: requests.models.Response = self.session.post("https://grok.com/c", multipart=mime)
            self.session.cookies.update(c_request.cookies)
            
            self.anon_user: str = Utils.between(c_request.text, '{"anonUserId":"', '"')
            self.c_run += 1
            
        else:
            
            match self.c_run:
                case 1:
                    data: str = dumps([{"anonUserId":self.anon_user}])
                case 2:
                    data: str = dumps([{"anonUserId":self.anon_user,**self.challenge_dict}])
            
            c_request: requests.models.Response = self.session.post('https://grok.com/c', data=data)
            self.session.cookies.update(c_request.cookies)

            match self.c_run:
                case 1:
                    start_idx = c_request.content.hex().find("3a6f38362c")
                    if start_idx != -1:
                        start_idx += len("3a6f38362c")
                        end_idx = c_request.content.hex().find("313a", start_idx)
                        if end_idx != -1:
                            challenge_hex = c_request.content.hex()[start_idx:end_idx]
                            challenge_bytes = bytes.fromhex(challenge_hex)

                    self.challenge_dict: dict = Anon.sign_challenge(challenge_bytes, self.keys["privateKey"])
                    Log.Success(f"Solved Challenge: {self.challenge_dict}")
                case 2:
                    self.verification_token, self.anim = Parser.get_anim(c_request.text, "grok-site-verification")
                    self.svg_data, self.numbers = Parser.parse_values(c_request.text, self.anim, self.xsid_script)
                    
            self.c_run += 1
        
    
    def start_convo(self, message: str, extra_data: dict = None) -> dict:
        
        if not extra_data:
            self._load()
            self.c_request(self.actions[0])
            self.c_request(self.actions[1])
            self.c_request(self.actions[2])
            xsid: str = Signature.generate_sign('/rest/app-chat/conversations/new', 'POST', self.verification_token, self.svg_data, self.numbers)
        else:
            self._load(extra_data)
            self.c_run: int = 1
            self.anon_user: str = extra_data["anon_user"]
            self.keys["privateKey"] = extra_data["privateKey"]
            self.c_request(self.actions[1])
            self.c_request(self.actions[2])
            xsid: str = Signature.generate_sign(f'/rest/app-chat/conversations/{extra_data["conversationId"]}/responses', 'POST', self.verification_token, self.svg_data, self.numbers)

        self.session.headers = self.headers.CONVERSATION
        self.session.headers.update({
            'baggage': self.baggage,
            'sentry-trace': f'{self.sentry_trace}-{str(uuid4()).replace("-", "")[:16]}-0',
            'x-statsig-id': xsid,
            'x-xai-request-id': str(uuid4()),
            'traceparent': f"00-{token_hex(16)}-{token_hex(8)}-00"
        })
        self.session.headers = Headers.fix_order(self.session.headers, self.headers.CONVERSATION)
        
        # Retry logic for conversation initiation
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not extra_data:
                    conversation_data: dict = {
                        'temporary': False,
                        'modelName': self.model,
                        'message': message,
                        'fileAttachments': [],
                        'imageAttachments': [],
                        'disableSearch': False,
                        'enableImageGeneration': True,
                        'returnImageBytes': False,
                        'returnRawGrokInXaiRequest': False,
                        'enableImageStreaming': True,
                        'imageGenerationCount': 2,
                        'forceConcise': False,
                        'toolOverrides': {},
                        'enableSideBySide': True,
                        'sendFinalMetadata': True,
                        'isReasoning': False,
                        'webpageUrls': [],
                        'disableTextFollowUps': False,
                        'responseMetadata': {
                            'requestModelDetails': {
                                'modelId': self.model,
                            },
                        },
                        'disableMemory': False,
                        'forceSideBySide': False,
                        'modelMode': self.model_mode,
                        'isAsyncChat': False,
                    }
                    
                    convo_request: requests.models.Response = self.session.post('https://grok.com/rest/app-chat/conversations/new', json=conversation_data, timeout=9999)
                else:
                    conversation_data: dict = {
                        'message': message,
                        'modelName': self.model,
                        'parentResponseId': extra_data["parentResponseId"],
                        'disableSearch': False,
                        'enableImageGeneration': True,
                        'imageAttachments': [],
                        'returnImageBytes': False,
                        'returnRawGrokInXaiRequest': False,
                        'fileAttachments': [],
                        'enableImageStreaming': True,
                        'imageGenerationCount': 2,
                        'forceConcise': False,
                        'toolOverrides': {},
                        'enableSideBySide': True,
                        'sendFinalMetadata': True,
                        'customPersonality': '',
                        'isReasoning': False,
                        'webpageUrls': [],
                        'metadata': {
                            'requestModelDetails': {
                                'modelId': self.model,
                            },
                            'request_metadata': {
                                'model': self.model,
                                'mode': self.mode,
                            },
                        },
                        'disableTextFollowUps': False,
                        'disableArtifact': False,
                        'isFromGrokFiles': False,
                        'disableMemory': False,
                        'forceSideBySide': False,
                        'modelMode': self.model_mode,
                        'isAsyncChat': False,
                        'skipCancelCurrentInflightRequests': False,
                        'isRegenRequest': False,
                    }

                    convo_request: requests.models.Response = self.session.post(f'https://grok.com/rest/app-chat/conversations/{extra_data["conversationId"]}/responses', json=conversation_data, timeout=9999)

                # Check if the status code indicates an error or if the response is valid
                if convo_request.status_code != 200:
                    return {"error": f"API Error {convo_request.status_code}: {convo_request.text}"}

                response = conversation_id = parent_response = image_urls = None
                stream_response: list = []
                
                # Split by newline and handle potential prefixes (like 0:{"..."})
                splits = convo_request.text.strip().split('\n')
                for line in splits:
                    try:
                        # Find the first '{' to strip prefixes like '0:' or '1:'
                        json_start = line.find('{')
                        if json_start != -1:
                            line = line[json_start:]
                        
                        data: dict = loads(line)
                    except Exception as e:
                        continue

                    # Targeted Extraction (Priority Path)
                    res = data.get('result', {})
                    
                    # 1. Capture IDs & Meta
                    cid = res.get('conversation', {}).get('conversationId')
                    if cid: conversation_id = cid

                    # 2. Capture Response Details (Direct or Nested)
                    # Use a broad set of keys to catch variations (modelResponse, message, text)
                    m_resp = res.get('response', {}).get('modelResponse') or res.get('modelResponse') or res
                    
                    if m_resp:
                        # Look for content in various possible keys
                        msg = m_resp.get('message') or m_resp.get('text')
                        
                        # Echo Prevention: Skip if the message is identical to our prompt (Echo)
                        # AI responses are usually much shorter than the full system prompt
                        if msg and isinstance(msg, str) and msg.strip() != message.strip():
                            # We only set response if it's the longest one we've seen (some chunks are partial)
                            if not response or len(msg) > len(response):
                                response = msg
                        
                        rid = m_resp.get('responseId')
                        if rid: parent_response = rid
                        
                        urls = m_resp.get('generatedImageUrls')
                        if urls: image_urls = urls

                    # 3. Stream Tokens (Incremental tokens)
                    token = res.get('response', {}).get('token') or res.get('token')
                    if token: stream_response.append(token)
                
                # FINAL FALLBACK: If 'response' is still None, but we have tokens, join them.
                if not response and stream_response:
                    response = "".join(stream_response)
                
                # If we successfully parsed a response or tokens, return them
                if response or stream_response:
                    return {
                        "response": response,
                        "stream_response": stream_response,
                        "images": image_urls,
                        "extra_data": {
                            "anon_user": self.anon_user,
                            "cookies": self.session.cookies.get_dict(),
                            "actions": self.actions,
                            "xsid_script": self.xsid_script,
                            "baggage": self.baggage,
                            "sentry_trace": self.sentry_trace,
                            "conversationId": conversation_id or (extra_data.get("conversationId") if extra_data else None),
                            "parentResponseId": parent_response,
                            "privateKey": self.keys["privateKey"]
                        }
                    }
                
                # If no response was found, check for specific "retryable" errors in the raw text
                if 'rejected by anti-bot rules' in convo_request.text:
                    Log.Error("Anti-bot rejection. Retrying...")
                    continue
                elif "Grok is under heavy usage right now" in convo_request.text:
                    Log.Error("Grok is overloaded. Retrying after delay...")
                    import time; time.sleep(3)
                    continue

                # Final fallback for error reporting
                if attempt == max_retries - 1:
                    return {"error": convo_request.text or f"Empty response with status {convo_request.status_code}"}

            except Exception as e:
                if attempt == max_retries - 1:
                    return {"error": str(e)}
                Log.Error(f"Grok request flake: {e}. Retrying...")
                import time; time.sleep(2)
        
        return {"error": "Max retries exceeded"}
            

