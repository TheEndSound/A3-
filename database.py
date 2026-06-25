#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SQLite 数据库持久化层。
线程安全，WAL 模式，自动建表。
"""

import json
import sqlite3
import threading
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "learning_system.db")

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _init_db(conn: sqlite3.Connection):
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_active TEXT NOT NULL DEFAULT (datetime('now')),
            message_count INTEGER NOT NULL DEFAULT 0,
            first_message_preview TEXT,
            last_message_preview TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('human', 'ai')),
            content TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            metadata TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS profiles (
            session_id TEXT PRIMARY KEY,
            knowledge_base TEXT,
            learning_style TEXT,
            weak_points TEXT,
            interest TEXT,
            learning_pace TEXT,
            interaction_summary TEXT,
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            overall_score INTEGER,
            knowledge_level TEXT,
            efficiency_level TEXT,
            weak_points_list TEXT,
            progress_summary TEXT,
            suggestions TEXT,
            score_data TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            content_type TEXT NOT NULL DEFAULT 'text',
            file_size INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, sequence);
        CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active DESC);
        CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id);
    """)


def _get_db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _init_db(_conn)
    return _conn


def upsert_session(session_id: str):
    with _lock:
        db = _get_db()
        db.execute(
            "INSERT INTO sessions (session_id, created_at, last_active) VALUES (?, datetime('now'), datetime('now')) "
            "ON CONFLICT(session_id) DO UPDATE SET last_active = datetime('now')",
            (session_id,),
        )
        db.commit()


def update_session_preview(session_id: str, role: str, content: str):
    with _lock:
        db = _get_db()
        preview = content[:100] if content else ""
        role_label = "human" if role == "human" else "ai"
        # Update message count and previews
        db.execute(
            "UPDATE sessions SET message_count = message_count + 1, last_active = datetime('now'), "
            "last_message_preview = ?, "
            "first_message_preview = COALESCE(first_message_preview, ?) "
            "WHERE session_id = ?",
            (preview, preview, session_id),
        )
        db.commit()


def insert_message(session_id: str, role: str, content: str, sequence: int, metadata: str | None = None):
    with _lock:
        db = _get_db()
        db.execute(
            "INSERT INTO messages (session_id, role, content, sequence, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (session_id, role, content, sequence, metadata),
        )
        db.commit()


def insert_evaluation(session_id: str, eval_data: dict):
    with _lock:
        db = _get_db()
        db.execute(
            "INSERT INTO evaluations (session_id, overall_score, knowledge_level, efficiency_level, "
            "weak_points_list, progress_summary, suggestions, score_data, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
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
        db.commit()


def list_sessions(limit: int = 20) -> list[dict]:
    with _lock:
        db = _get_db()
        rows = db.execute(
            "SELECT session_id, created_at, last_active, message_count, first_message_preview, last_message_preview "
            "FROM sessions ORDER BY last_active DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "session_id": r[0],
            "created_at": r[1],
            "last_active": r[2],
            "message_count": r[3],
            "first_message_preview": r[4],
            "last_message_preview": r[5],
        }
        for r in rows
    ]


def get_session(session_id: str) -> dict | None:
    with _lock:
        db = _get_db()
        row = db.execute(
            "SELECT session_id, created_at, last_active, message_count, first_message_preview, last_message_preview "
            "FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "session_id": row[0],
        "created_at": row[1],
        "last_active": row[2],
        "message_count": row[3],
        "first_message_preview": row[4],
        "last_message_preview": row[5],
    }


def get_session_messages(session_id: str) -> list[dict]:
    with _lock:
        db = _get_db()
        rows = db.execute(
            "SELECT role, content, sequence, created_at, metadata "
            "FROM messages WHERE session_id = ? ORDER BY sequence ASC",
            (session_id,),
        ).fetchall()
    return [
        {
            "role": r[0],
            "content": r[1],
            "sequence": r[2],
            "created_at": r[3],
            "metadata": r[4],
        }
        for r in rows
    ]


def save_profile(session_id: str, profile: dict):
    with _lock:
        db = _get_db()
        db.execute(
            "INSERT INTO profiles (session_id, knowledge_base, learning_style, weak_points, "
            "interest, learning_pace, interaction_summary, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "knowledge_base = excluded.knowledge_base, "
            "learning_style = excluded.learning_style, "
            "weak_points = excluded.weak_points, "
            "interest = excluded.interest, "
            "learning_pace = excluded.learning_pace, "
            "interaction_summary = excluded.interaction_summary, "
            "updated_at = datetime('now')",
            (
                session_id,
                profile.get("knowledge_base"),
                profile.get("learning_style"),
                json.dumps(profile.get("weak_points", []), ensure_ascii=False),
                profile.get("interest"),
                profile.get("learning_pace"),
                profile.get("interaction_summary"),
            ),
        )
        db.commit()


def get_profile(session_id: str) -> dict | None:
    with _lock:
        db = _get_db()
        row = db.execute(
            "SELECT knowledge_base, learning_style, weak_points, interest, "
            "learning_pace, interaction_summary "
            "FROM profiles WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    weak_points = json.loads(row[2]) if row[2] else []
    return {
        "knowledge_base": row[0],
        "learning_style": row[1],
        "weak_points": weak_points,
        "interest": row[3],
        "learning_pace": row[4],
        "interaction_summary": row[5],
    }


def delete_session(session_id: str):
    with _lock:
        db = _get_db()
        db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        db.commit()


def insert_document(session_id: str, title: str, content: str, content_type: str = "text", file_size: int = 0) -> int:
    with _lock:
        db = _get_db()
        cur = db.execute(
            "INSERT INTO documents (session_id, title, content, content_type, file_size, created_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (session_id, title, content, content_type, file_size),
        )
        db.commit()
        return cur.lastrowid


def list_documents(session_id: str) -> list[dict]:
    with _lock:
        db = _get_db()
        rows = db.execute(
            "SELECT id, session_id, title, content_type, file_size, created_at "
            "FROM documents WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
    return [
        {
            "id": r[0],
            "session_id": r[1],
            "title": r[2],
            "content_type": r[3],
            "file_size": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]


def get_document(doc_id: int) -> dict | None:
    with _lock:
        db = _get_db()
        row = db.execute(
            "SELECT id, session_id, title, content, content_type, file_size, created_at "
            "FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "session_id": row[1],
        "title": row[2],
        "content": row[3],
        "content_type": row[4],
        "file_size": row[5],
        "created_at": row[6],
    }


def get_documents_content(session_id: str) -> list[dict]:
    """获取会话所有文档的完整内容（用于上下文拼接）。"""
    with _lock:
        db = _get_db()
        rows = db.execute(
            "SELECT id, title, content, content_type FROM documents WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
    return [
        {"id": r[0], "title": r[1], "content": r[2], "content_type": r[3]}
        for r in rows
    ]


def delete_document(doc_id: int):
    with _lock:
        db = _get_db()
        db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        db.commit()
