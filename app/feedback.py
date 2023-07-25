import os
import sys
import logging
import datetime
import sqlite3


class Feedback:
    logger = logging.getLogger(__name__)
    kFeedbackFile = os.path.join('feedback', 'feedback.sqlite3.db')
    kSchemaFile = os.path.join('feedback', 'schema.sql')

    def __init__(self, feedback_file=None):
        self._feedback_file = feedback_file if feedback_file else Feedback.kFeedbackFile

    def get_db_connection(self):
        conn = sqlite3.connect(self._feedback_file)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.get_db_connection() as con:
            with open(self.kSchemaFile, mode='r') as fp:
                con.cursor().executescript(fp.read())
            con.commit()

    def save(self, sms_text, model_key=None, pred_class=None, feedback_positive=None, feedback_class=None, feedback_text=None):
        debug_msg = f'Saving user feedback to {self._feedback_file}'
        timestamp = datetime.datetime.utcnow().isoformat()

        try:
            with self.get_db_connection() as conn:
                conn.execute(
                    'INSERT INTO feedback (timestamp, sms_text, model_key, pred_class, feedback_positive, feedback_class, feedback_text) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (timestamp, sms_text, model_key, pred_class, feedback_positive, feedback_class, feedback_text)
                )
                conn.commit()
            return True
        except Exception as e:
            self.logger.error(f'{debug_msg}: status=failed to save')
        return False