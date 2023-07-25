CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    sms_text TEXT,
    model_key TEXT,
    pred_class TEXT,
    feedback_positive INTEGER NOT NULL,
    feedback_class TEXT,
    feedback_text TEXT
)