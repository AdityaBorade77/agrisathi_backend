import sqlite3
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path("chat_history.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            role TEXT,
            msg_type TEXT,
            content_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized.")

def create_chat(title: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO chats (title) VALUES (?)', (title,))
    chat_id = c.lastrowid
    conn.commit()
    conn.close()
    return chat_id

def get_chats() -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT id, title, created_at FROM chats ORDER BY created_at DESC')
    chats = [dict(row) for row in c.fetchall()]
    conn.close()
    return chats

def get_chat_messages(chat_id: int) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT id, role, msg_type, content_json, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC', (chat_id,))
    messages = []
    for row in c.fetchall():
        msg = dict(row)
        msg['content'] = json.loads(msg['content_json'])
        del msg['content_json']
        messages.append(msg)
    conn.close()
    return messages

def add_message(chat_id: int, role: str, msg_type: str, content: dict) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO messages (chat_id, role, msg_type, content_json)
        VALUES (?, ?, ?, ?)
    ''', (chat_id, role, msg_type, json.dumps(content)))
    msg_id = c.lastrowid
    conn.commit()
    conn.close()
    return msg_id

def delete_chat(chat_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('PRAGMA foreign_keys = ON')
    c.execute('DELETE FROM chats WHERE id = ?', (chat_id,))
    conn.commit()
    conn.close()
