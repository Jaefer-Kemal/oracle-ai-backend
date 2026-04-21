import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add backend to path so we can import our core utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils.browser import get_session_cookies

# Configuration
# This script should be run locally. It needs the DATABASE_URL of your production/staging DB.
DATABASE_URL = os.getenv("DATABASE_URL")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("token-pusher")

def push_tokens():
    if not DATABASE_URL:
        logger.error("DATABASE_URL not found in environment. Please set it before running.")
        return

    logger.info("Connecting to database...")
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        # 1. Fetch Gemini Cookies
        logger.info("Attempting to fetch Gemini cookies from local browser...")
        gemini_cookies = get_session_cookies("gemini", browser="chrome") # Try chrome first
        
        psid = gemini_cookies.get("__Secure-1PSID")
        psidts = gemini_cookies.get("__Secure-1PSIDTS")
        
        if psid and psidts:
            logger.info("Found Gemini cookies. Pushing to DB...")
            db.execute(text("UPDATE app_settings SET value = :val WHERE key = 'gemini_1psid'"), {"val": psid})
            db.execute(text("UPDATE app_settings SET value = :val WHERE key = 'gemini_1psidts'"), {"val": psidts})
        else:
            # Try Firefox fallback
            logger.info("Chrome cookies not found. Trying Firefox...")
            gemini_cookies = get_session_cookies("gemini", browser="firefox")
            psid = gemini_cookies.get("__Secure-1PSID")
            psidts = gemini_cookies.get("__Secure-1PSIDTS")
            if psid and psidts:
                db.execute(text("UPDATE app_settings SET value = :val WHERE key = 'gemini_1psid'"), {"val": psid})
                db.execute(text("UPDATE app_settings SET value = :val WHERE key = 'gemini_1psidts'"), {"val": psidts})
            else:
                logger.warning("Could not find Gemini cookies in any browser.")

        # 2. Fetch Grok Cookies (If applicable)
        # Add Grok specific pushes here if needed
        
        db.commit()
        logger.info("Successfully synchronized session tokens from local browser to cloud database.")
        
    except Exception as e:
        logger.error(f"Faiure during token push: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    push_tokens()
