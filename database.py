#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PostgreSQL 数据库持久化层。
线程安全，连接池，自动建表。
"""

import json
import threading
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.pool
from crypto_utils import get_env_secret

# ================== 数据库连接配置 ==================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "ai_learning"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": get_env_secret("DB_PASSWORD", "123456"),
}

_lock = threading.Lock()
_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            **DB_CONFIG,
        )
        _init_db()
    return _pool


@contextmanager
def _get_conn():
    """获取数据库连接上下文管理器，自动提交/回滚/归还。"""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ================== 建表 ==================

_DDL_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS sessions (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(256) UNIQUE NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        last_active TIMESTAMP NOT NULL DEFAULT NOW(),
        message_count INTEGER NOT NULL DEFAULT 0,
        first_message_preview TEXT,
        last_message_preview TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS messages (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(256) NOT NULL,
        role VARCHAR(16) NOT NULL CHECK(role IN ('human', 'ai')),
        content TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        metadata TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    )""",
    """CREATE TABLE IF NOT EXISTS profiles (
        session_id VARCHAR(256) PRIMARY KEY,
        knowledge_base VARCHAR(32),
        learning_style VARCHAR(32),
        cognitive_style VARCHAR(32),
        weak_points JSONB DEFAULT '[]',
        error_patterns JSONB DEFAULT '[]',
        interest TEXT,
        learning_pace VARCHAR(16),
        learning_goals TEXT,
        motivation_level VARCHAR(16),
        interaction_summary TEXT,
        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    )""",
    """CREATE TABLE IF NOT EXISTS evaluations (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(256) NOT NULL,
        overall_score INTEGER,
        knowledge_level TEXT,
        efficiency_level TEXT,
        weak_points_list TEXT,
        progress_summary TEXT,
        suggestions TEXT,
        score_data JSONB NOT NULL DEFAULT '{}',
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    )""",
    """CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(256) NOT NULL,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        content_type VARCHAR(32) NOT NULL DEFAULT 'text',
        file_size INTEGER DEFAULT 0,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    )""",
    "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, sequence)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active DESC)",
    """CREATE TABLE IF NOT EXISTS learning_plans (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(256) NOT NULL,
        diagnosis TEXT,
        summary TEXT,
        steps JSONB NOT NULL DEFAULT '[]',
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    )""",
    "CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_learning_plans_session ON learning_plans(session_id)",
]


def _init_db():
    with _lock:
        pool = _get_pool()
        conn = pool.getconn()
        try:
            conn.autocommit = True
            cur = conn.cursor()
            for stmt in _DDL_STATEMENTS:
                cur.execute(stmt)
        finally:
            conn.autocommit = False
            pool.putconn(conn)


# ================== 会话管理 ==================

def upsert_session(session_id: str):
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (session_id, created_at, last_active) "
            "VALUES (%s, NOW(), NOW()) "
            "ON CONFLICT(session_id) DO UPDATE SET last_active = NOW()",
            (session_id,),
        )


def update_session_preview(session_id: str, role: str, content: str):
    with _get_conn() as conn:
        cur = conn.cursor()
        preview = content[:100] if content else ""
        cur.execute(
            "UPDATE sessions SET message_count = message_count + 1, last_active = NOW(), "
            "last_message_preview = %s, "
            "first_message_preview = COALESCE(first_message_preview, %s) "
            "WHERE session_id = %s",
            (preview, preview, session_id),
        )


def insert_message(session_id: str, role: str, content: str, sequence: int, metadata: str | None = None):
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages (session_id, role, content, sequence, metadata, created_at) "
            "VALUES (%s, %s, %s, %s, %s, NOW())",
            (session_id, role, content, sequence, metadata),
        )


def list_sessions(limit: int = 20) -> list[dict]:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT session_id, created_at, last_active, message_count, "
            "first_message_preview, last_message_preview "
            "FROM sessions ORDER BY last_active DESC LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
    return [
        {
            "session_id": r[0],
            "created_at": r[1].isoformat() if hasattr(r[1], 'isoformat') else str(r[1]),
            "last_active": r[2].isoformat() if hasattr(r[2], 'isoformat') else str(r[2]),
            "message_count": r[3],
            "first_message_preview": r[4],
            "last_message_preview": r[5],
        }
        for r in rows
    ]


def get_session(session_id: str) -> dict | None:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT session_id, created_at, last_active, message_count, "
            "first_message_preview, last_message_preview "
            "FROM sessions WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "session_id": row[0],
        "created_at": row[1].isoformat() if hasattr(row[1], 'isoformat') else str(row[1]),
        "last_active": row[2].isoformat() if hasattr(row[2], 'isoformat') else str(row[2]),
        "message_count": row[3],
        "first_message_preview": row[4],
        "last_message_preview": row[5],
    }


def get_session_messages(session_id: str) -> list[dict]:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content, sequence, created_at, metadata "
            "FROM messages WHERE session_id = %s ORDER BY sequence ASC",
            (session_id,),
        )
        rows = cur.fetchall()
    return [
        {
            "role": r[0],
            "content": r[1],
            "sequence": r[2],
            "created_at": r[3].isoformat() if hasattr(r[3], 'isoformat') else str(r[3]),
            "metadata": r[4],
        }
        for r in rows
    ]


def delete_session(session_id: str):
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))


# ================== 画像管理 (10 维度) ==================

def save_profile(session_id: str, profile: dict):
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO profiles (session_id, knowledge_base, learning_style, cognitive_style, "
            "weak_points, error_patterns, interest, learning_pace, learning_goals, "
            "motivation_level, interaction_summary, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "knowledge_base = EXCLUDED.knowledge_base, "
            "learning_style = EXCLUDED.learning_style, "
            "cognitive_style = EXCLUDED.cognitive_style, "
            "weak_points = EXCLUDED.weak_points, "
            "error_patterns = EXCLUDED.error_patterns, "
            "interest = EXCLUDED.interest, "
            "learning_pace = EXCLUDED.learning_pace, "
            "learning_goals = EXCLUDED.learning_goals, "
            "motivation_level = EXCLUDED.motivation_level, "
            "interaction_summary = EXCLUDED.interaction_summary, "
            "updated_at = NOW()",
            (
                session_id,
                profile.get("knowledge_base"),
                profile.get("learning_style"),
                profile.get("cognitive_style"),
                json.dumps(profile.get("weak_points", []), ensure_ascii=False),
                json.dumps(profile.get("error_patterns", []), ensure_ascii=False),
                profile.get("interest"),
                profile.get("learning_pace"),
                profile.get("learning_goals"),
                profile.get("motivation_level"),
                profile.get("interaction_summary"),
            ),
        )


def get_profile(session_id: str) -> dict | None:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT knowledge_base, learning_style, cognitive_style, weak_points, "
            "error_patterns, interest, learning_pace, learning_goals, "
            "motivation_level, interaction_summary "
            "FROM profiles WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "knowledge_base": row[0],
        "learning_style": row[1],
        "cognitive_style": row[2],
        "weak_points": row[3] if isinstance(row[3], list) else (json.loads(row[3]) if row[3] else []),
        "error_patterns": row[4] if isinstance(row[4], list) else (json.loads(row[4]) if row[4] else []),
        "interest": row[5],
        "learning_pace": row[6],
        "learning_goals": row[7],
        "motivation_level": row[8],
        "interaction_summary": row[9],
    }


# ================== 评估管理 ==================

def insert_evaluation(session_id: str, eval_data: dict):
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO evaluations (session_id, overall_score, knowledge_level, "
            "efficiency_level, weak_points_list, progress_summary, suggestions, "
            "score_data, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())",
            (
                session_id,
                eval_data.get("overall_score"),
                eval_data.get("knowledge_level"),
                eval_data.get("efficiency_level"),
                eval_data.get("weak_points_list"),
                eval_data.get("progress_summary"),
                eval_data.get("suggestions"),
                json.dumps(eval_data, ensure_ascii=False),
            ),
        )


# ================== 文档管理 ==================

def insert_document(session_id: str, title: str, content: str, content_type: str = "text", file_size: int = 0) -> int:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (session_id, title, content, content_type, file_size, created_at) "
            "VALUES (%s, %s, %s, %s, %s, NOW()) RETURNING id",
            (session_id, title, content, content_type, file_size),
        )
        return cur.fetchone()[0]


def list_documents(session_id: str) -> list[dict]:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, session_id, title, content_type, file_size, created_at "
            "FROM documents WHERE session_id = %s ORDER BY created_at DESC",
            (session_id,),
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "session_id": r[1],
            "title": r[2],
            "content_type": r[3],
            "file_size": r[4],
            "created_at": r[5].isoformat() if hasattr(r[5], 'isoformat') else str(r[5]),
        }
        for r in rows
    ]


def get_document(doc_id: int) -> dict | None:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, session_id, title, content, content_type, file_size, created_at "
            "FROM documents WHERE id = %s",
            (doc_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "session_id": row[1],
        "title": row[2],
        "content": row[3],
        "content_type": row[4],
        "file_size": row[5],
        "created_at": row[6].isoformat() if hasattr(row[6], 'isoformat') else str(row[6]),
    }


def get_documents_content(session_id: str) -> list[dict]:
    """获取会话所有文档的完整内容（用于上下文拼接）。"""
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, content, content_type FROM documents "
            "WHERE session_id = %s ORDER BY created_at DESC",
            (session_id,),
        )
        rows = cur.fetchall()
    return [
        {"id": r[0], "title": r[1], "content": r[2], "content_type": r[3]}
        for r in rows
    ]


def delete_document(doc_id: int):
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))


# ================== 学习路径规划 ==================

def save_plan(session_id: str, diagnosis: str, summary: str, steps: list):
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO learning_plans (session_id, diagnosis, summary, steps, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, NOW(), NOW())",
            (session_id, diagnosis, summary, json.dumps(steps, ensure_ascii=False)),
        )


def get_plan(session_id: str) -> dict | None:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT diagnosis, summary, steps, created_at, updated_at "
            "FROM learning_plans WHERE session_id = %s "
            "ORDER BY updated_at DESC LIMIT 1",
            (session_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "diagnosis": row[0],
        "summary": row[1],
        "steps": row[2] if isinstance(row[2], list) else (json.loads(row[2]) if row[2] else []),
        "created_at": row[3].isoformat() if hasattr(row[3], 'isoformat') else str(row[3]),
        "updated_at": row[4].isoformat() if hasattr(row[4], 'isoformat') else str(row[4]),
    }
