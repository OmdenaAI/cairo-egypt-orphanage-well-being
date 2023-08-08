from django.db import connection
import pickle
import base64


class TakeAttendance:
    def __init__(self):
        self.classNames = []
        self.encodeListKnown = []
        self.fetch_encods()

    def run_query(self, query):
        cursor = connection.cursor()
        cursor.execute(query)

        return cursor

    def convert(self, encoding, name):
        np_bytes = base64.b64decode(encoding)
        np_array = pickle.loads(np_bytes)
        self.encodeListKnown.append(np_array)
        self.classNames.append(name)

    def fetch_encods(self):
        record_list = self.run_query('''
        SELECT * FROM webapp_orphan_list
        ''').fetchall()

        for i in range(len(record_list)):
            name = record_list[i][1]
            encoding1 = record_list[i][5]
            encoding2 = record_list[i][6]
            encoding3 = record_list[i][7]

            if encoding1:
                self.convert(encoding1, name)

            if encoding2:
                self.convert(encoding2, name)

            if encoding3:
                self.convert(encoding3, name)

