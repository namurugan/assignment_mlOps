import os
import logging
from logging.handlers import TimedRotatingFileHandler
import sqlite3
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DB_PATH = os.path.join(LOG_DIR, "logs.db")
os.makedirs(LOG_DIR, exist_ok=True)

class SQLiteHandler(logging.Handler):
    def __init__(self, db_path=DB_PATH):
        super().__init__()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                message TEXT
            )
        """)
        self.conn.commit()

    def emit(self, record):
        timestamp = datetime.now().isoformat()
        message = self.format(record)
        level = record.levelname

        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO logs (timestamp, level, message) VALUES (?, ?, ?)",
            (timestamp, level, message)
        )
        self.conn.commit()

def get_logger():
    logger = logging.getLogger("prediction_logger")
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(LOG_DIR, f"{today}.log")

    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        backupCount=7,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)  # Also set handlers to DEBUG

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(formatter)

    sqlite_handler = SQLiteHandler()
    sqlite_handler.setLevel(logging.DEBUG)  # Make sure DB handler also logs all levels
    sqlite_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(sqlite_handler)

    return logger
