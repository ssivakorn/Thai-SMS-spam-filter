import os
import sys
import logging
import datetime
import sqlite3


class Feedback:
    logger = logging.getLogger(__name__)
    kFeedbackFile = os.path.join('feedback', 'feedback.sqlite3.db')

    def __init__(self, feedback_file=None):
        self._feedback_file = feedback_file if feedback_file else Feedback.kFeedbackFile

    def save(self, sms_text, model=None, pred_class=None, feedback_positive=None, feedback_class=None, feedback_text=None):
        debug_msg = f'Saving user feedback to {self._feedback_file}'
        current_time = datetime.datetime.utcnow()

        feedback_data = (current_time,
                         sms_text,
                         model,
                         pred_class,
                         feedback_positive,
                         feedback_class,
                         feedback_text)

        # try:
        #     con = sqlite3.connect(self._feedback_file)
        #     cur = con.cursor()
        #     cur.execute('INSERT INTO feedback VALUES(?, ?, ?, ?, ?, ?)', feedback_data)
        #     con.commit()
        #     self.logger.info(f'{debug_msg}, status=successful')
        #     return True
        # except Exception as e:
        #     self.logger.error(f'{debug_msg}: status=failed, reason={e}')
        # return False

        print(feedback_data)
        return True