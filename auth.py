import os
import bcrypt
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_REFRESH_SECRET = os.getenv("JWT_REFRESH_SECRET")

if not JWT_SECRET or not JWT_REFRESH_SECRET:
    raise ValueError("FATAL ERROR: JWT_SECRET and JWT_REFRESH_SECRET must be explicitly set in the .env file for security purposes.")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour access tokens
REFRESH_TOKEN_EXPIRE_DAYS = 30    # 30-day refresh tokens

def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt directly to avoid passlib bugs."""
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password using bcrypt directly."""
    try:
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)

def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, JWT_REFRESH_SECRET, algorithm=ALGORITHM)

def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            return None
        return payload
    except JWTError:
        return None

def decode_refresh_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, JWT_REFRESH_SECRET, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            return None
        return payload
    except JWTError:
        return None
