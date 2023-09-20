import psycopg2
import datetime

# TODO: retrieve database configs from .env file
DB_USER="postgres"
DB_PASSWORD="root"
DB_HOST="127.0.0.1"
DB_PORT="5432"
DB_NAME = "orphansdb"

TABLE_NAME = "mlpipeline_detection"


def insert_prediction(mood_name:str, activity_name:str, reference_video:str, recorded_date:datetime.datetime, 
                      camera_id:int, profile_id:int)->None:
    """
        Insert models' predictions in the database

        Usage eg:
                current_date = datetime.datetime.today()
                insert_prediction(mood_name="happy", activity_name="playing", reference_video="xxx", \
                    recorded_date=current_date, camera_id=2, profile_id=3)
    """

    try:
        connection = psycopg2.connect(user=DB_USER,
                                      password=DB_PASSWORD,
                                      host=DB_HOST,
                                      port=DB_PORT,
                                      database=DB_NAME)
        cursor = connection.cursor()

        postgres_insert_query =f"""INSERT INTO {TABLE_NAME}(mood_name, activity_name,reference_video,recorded_date, camera_id, profile_id) VALUES (%s, %s,%s,%s, %s, %s)""" #  #""" INSERT INTO mobile (ID, MODEL, PRICE) VALUES (%s,%s,%s)"""
        record_to_insert = (mood_name, activity_name,reference_video,recorded_date, camera_id, profile_id)#(5, 'One Plus 6', 950)
        cursor.execute(postgres_insert_query, record_to_insert)

        connection.commit()
        count = cursor.rowcount
        print(count, f"Record inserted successfully into {TABLE_NAME} table")

    except (Exception, psycopg2.Error) as error:
        print(f"Failed to insert record into {TABLE_NAME} table", error)

    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

