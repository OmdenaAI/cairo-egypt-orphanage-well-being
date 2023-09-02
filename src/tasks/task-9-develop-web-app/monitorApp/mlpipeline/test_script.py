import psycopg2
# from config import config

try:
    # params = config()
    connection = psycopg2.connect(
        host="localhost",
        database="orphansdb",
        user="postgres",
        password="12345")
    cursor = connection.cursor()
    cursor.execute('SELECT * from mlpipeline_scriptexecutions order by exec_start_time desc limit 1;')
    result = cursor.fetchone()
    while result[2] == 'Running':
        cursor.execute('SELECT * from mlpipeline_scriptexecutions order by exec_start_time desc limit 1;')
        result = cursor.fetchone()
    print(result[2])
except Exception as e:
    print(e)
