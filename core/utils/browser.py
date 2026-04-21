import logging
import browser_cookie3
import platform
import os
import sqlite3
import json
import base64
from pathlib import Path
from typing import Optional, Literal, Dict, Any

# Windows-specific imports for cookie decryption
if platform.system().lower() == "windows":
    try:
        import win32crypt
        from Crypto.Cipher import AES
        HAS_CRYPTO = True
    except ImportError:
        HAS_CRYPTO = False
else:
    HAS_CRYPTO = False

logger = logging.getLogger("rag-backend")

class CrossPlatformCookieExtractor:
    """Enhanced cookie extractor for Oracle AI (Ported from WebAI-to-API)"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.is_windows = self.system == "windows"
        logger.info(f"Initialized cookie extractor for {self.system}")
    
    def _get_browser_profile_paths(self, browser_name: str) -> Dict[str, Any]:
        """Get browser profile paths for different operating systems"""
        paths = {}
        
        if self.is_windows:
            user_data = os.path.expanduser("~")
            if browser_name == "chrome":
                base_path = os.path.join(user_data, "AppData", "Local", "Google", "Chrome", "User Data")
                possible_paths = [
                    os.path.join(base_path, "Default", "Network", "Cookies"),
                    os.path.join(base_path, "Default", "Cookies"),
                ]
                cookies_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        cookies_path = path
                        break
                paths = {"cookies_db": cookies_path, "local_state": os.path.join(base_path, "Local State")}
                
            elif browser_name == "brave":
                base_path = os.path.join(user_data, "AppData", "Local", "BraveSoftware", "Brave-Browser", "User Data")
                possible_paths = [
                    os.path.join(base_path, "Default", "Network", "Cookies"),
                    os.path.join(base_path, "Default", "Cookies"),
                ]
                cookies_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        cookies_path = path
                        break
                paths = {"cookies_db": cookies_path, "local_state": os.path.join(base_path, "Local State")}
                
            elif browser_name == "edge":
                base_path = os.path.join(user_data, "AppData", "Local", "Microsoft", "Edge", "User Data")
                possible_paths = [
                    os.path.join(base_path, "Default", "Network", "Cookies"),
                    os.path.join(base_path, "Default", "Cookies"),
                ]
                cookies_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        cookies_path = path
                        break
                paths = {"cookies_db": cookies_path, "local_state": os.path.join(base_path, "Local State")}
                
            elif browser_name == "firefox":
                firefox_path = os.path.join(user_data, "AppData", "Roaming", "Mozilla", "Firefox", "Profiles")
                if os.path.exists(firefox_path):
                    profiles = [d for d in os.listdir(firefox_path) if os.path.isdir(os.path.join(firefox_path, d))]
                    if profiles:
                        profile_path = os.path.join(firefox_path, profiles[0])
                        paths = {"cookies_db": os.path.join(profile_path, "cookies.sqlite")}
        
        return paths
    
    def _decrypt_chrome_cookie_value(self, encrypted_value: bytes, local_state_path: str) -> Optional[str]:
        if not self.is_windows or not HAS_CRYPTO: return None
        try:
            with open(local_state_path, 'r', encoding='utf-8') as f:
                local_state = json.load(f)
            encrypted_key = base64.b64decode(local_state['os_crypt']['encrypted_key'])[5:]
            key = win32crypt.CryptUnprotectData(encrypted_key, None, None, None, 0)[1]
            
            version = encrypted_value[:3]
            if version != b'v10' and version != b'v11':
                return win32crypt.CryptUnprotectData(encrypted_value, None, None, None, 0)[1].decode('utf-8')
            
            nonce = encrypted_value[3:15]
            ciphertext = encrypted_value[15:-16]
            tag = encrypted_value[-16:]
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
            return cipher.decrypt_and_verify(ciphertext, tag).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to decrypt Chrome cookie: {e}")
            return None

    def get_cookies(self, browser_name: str, domain_filter: str) -> Dict[str, str]:
        """Generic cookie getter avec retry logic"""
        logger.info(f"Fetching cookies for {domain_filter} from {browser_name}")
        results = {}
        
        # Try browser_cookie3 first
        try:
            cj = None
            if browser_name == "firefox": cj = browser_cookie3.firefox(domain_name=domain_filter)
            elif browser_name == "chrome": cj = browser_cookie3.chrome(domain_name=domain_filter)
            elif browser_name == "edge": cj = browser_cookie3.edge(domain_name=domain_filter)
            elif browser_name == "brave": cj = browser_cookie3.brave(domain_name=domain_filter)
            
            if cj:
                for cookie in cj:
                    results[cookie.name] = cookie.value
        except Exception as e:
            logger.warning(f"browser_cookie3 fallback triggered for {browser_name}: {e}")

        # If empty, try manual DB extraction (often needed on Windows)
        if not results and self.is_windows:
            paths = self._get_browser_profile_paths(browser_name)
            if "cookies_db" in paths:
                db_path = paths["cookies_db"]
                local_state = paths.get("local_state")
                
                import tempfile, shutil
                with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
                    shutil.copy2(db_path, tmp.name)
                    try:
                        conn = sqlite3.connect(tmp.name)
                        cursor = conn.cursor()
                        # Chromium style
                        cursor.execute("SELECT name, value, encrypted_value FROM cookies WHERE host_key LIKE ?", (f'%{domain_filter}%',))
                        for name, val, enc_val in cursor.fetchall():
                            if not val and enc_val and local_state:
                                val = self._decrypt_chrome_cookie_value(enc_val, local_state)
                            if val: results[name] = val
                        conn.close()
                    except: pass
                    finally: os.unlink(tmp.name)
        
        return results

def get_session_cookies(service: Literal["gemini", "grok"], browser: str = "chrome") -> Dict[str, str]:
    extractor = CrossPlatformCookieExtractor()
    domain = "google.com" if service == "gemini" else "grok.com"
    all_cookies = extractor.get_cookies(browser, domain)
    
    if service == "gemini":
        return {
            "__Secure-1PSID": all_cookies.get("__Secure-1PSID", ""),
            "__Secure-1PSIDTS": all_cookies.get("__Secure-1PSIDTS", "")
        }
    else: # grok
        # Grok usually uses 'sso' or specific local storage, 
        # but web-version uses cookies like 'auth_token' or similar depending on the reverse logic.
        # We'll capture what we find.
        return all_cookies
