# 🛡️ Oracle AI Knowledge Base - Backend

[![Python Version](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Database](https://img.shields.io/badge/Database-PostgreSQL-336791?style=flat&logo=postgresql)](https://www.postgresql.org/)
[![Vector](https://img.shields.io/badge/Vector-pgvector-FF6F00?style=flat)](https://github.com/pgvector/pgvector)
[![RAG](https://img.shields.io/badge/RAG-Cohere%20%2B%20Grok-blue?style=flat)](https://cohere.ai)

A high-performance enterprise RAG (Retrieval-Augmented Generation) infrastructure. This backend orchestrates complex document processing, manages semantic vector storage, and provides a secure API for AI-powered intelligence.

> **🙏 Special Acknowledgement & Credit**
>
> The Grok integration in this project utilizes the reverse-engineered logic from the incredible [Grok-Api by realasfngl](https://github.com/realasfngl/Grok-Api). 
> 
> *The original Grok-Api is a free wrapper that allows you to interact with Grok's conversational AI without requiring official API access. It includes streaming support, proxy handling, and multi-worker deployment features. If you are looking for the standalone API, please check out and star their repository!*

---

## 🏗️ Core Architecture

The system is designed with scalability and performance in mind, utilizing a modern AI stack.

- **Semantic Intelligence Layer**: Uses **Cohere** for high-accuracy text embeddings and **Grok** for state-of-the-art generative answers.
- **Persistence Layer**: PostgreSQL with `pgvector` for efficient similarity searches and relational metadata.
- **Service Layer**: Decoupled AI services for easy swapping of models or providers.
- **Security Layer**: Stateless JWT authentication with secure cookie-based refresh flows.

---

## ✨ Key Features

- **📂 Enterprise RAG**: Advanced file parsing and chunking with semantic retrieval.
- **⚡ Dual-Engine AI**: Optimized pipeline using Cohere for "v3" embeddings and Grok for ultra-fast reasoning.
- **🔐 Secure RBAC**: Strict administrative controls with protected API routes.
- **💾 Soft-Purge Management**: Soft-delete logic for audit trails and "Trash" restoration capabilities.
- **📊 Real-time Audit**: capturing WHO, WHAT, and WHEN for every critical system interaction.
- **🛠️ Self-Healing Auth**: Automatic migration of environment-configured credentials to secure database storage on first boot.

---

## 🛠️ Technical Stack

- **Language**: Python 3.10+ for rapid AI development.
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) for asynchronous high-performance routing.
- **Database**: [PostgreSQL (Neon)](https://neon.tech/) with `pgvector` for vector storage.
- **ORM**: SQLAlchemy for robust database management.
- **AI Services**: [Cohere](https://cohere.ai) (Embeddings) & [Grok](https://x.ai) (Generation).
- **Security**: JWT (Access/Refresh), Bcrypt for password hashing.

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python** (3.10+)
- **Poetry** (Package Manager)
- **PostgreSQL** (with `pgvector` extension enabled)

### 2. Environment Configuration
Create a `.env` file at the `/backend` root:

```bash
# Database (Neon/Postgres)
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require

# AI API Keys
COHERE_API_KEY=your_cohere_key_here
# Note: Cohere provides a free "Trial" API key that supports 100 requests/min.
# This codebase is optimized for this limit, ensuring high performance at zero cost.

# Grok-Api is handled via reverse-engineering (no key required)

# Security
JWT_SECRET=your_super_secret_key_here
JWT_REFRESH_SECRET=your_refresh_secret_key_here

# Initial Admin Seeding (One-time process)
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_initial_password
ADMIN_DISPLAY_NAME=Administrator
```

---

## 🏃 Operation Commands

### 📦 Dependency Setup
Install project dependencies using Poetry:

```bash
poetry install
```

### 🔥 Start Server
Launch the FastAPI development server:

```bash
poetry run python main.py
```

---

## 🧪 Development & Quality

### 🔐 Initial Admin Account Setup
The system features a "Self-Healing" seeder. On the very first login, the system will check the `ADMIN_USERNAME` and `ADMIN_PASSWORD` from your `.env` file. If they are valid, it will automatically hash them and migrate them into secure database storage. 

> [!IMPORTANT]
> Once the first login is successful and the account is seeded, the system will use the database record. You can then change your credentials via the Admin Dashboard.

### API Exploration (Swagger)
The live interactive documentation is available at:
**`http://localhost:8000/docs`**

---

## 📂 Repository Structure

```text
├── core/               # Reverse-engineered AI engine (Grok-Api)
├── auth.py             # JWT & Password security logic
├── models.py           # Database schemas & SQLAlchemy models
├── services.py         # AI provider implementations (Cohere/Grok)
├── main.py             # FastAPI entry point & API routes
└── .env                # Local secrets (never committed)
```
