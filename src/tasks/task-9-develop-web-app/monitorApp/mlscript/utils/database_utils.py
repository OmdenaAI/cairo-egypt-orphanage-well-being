import psycopg2
import datetime
from contextlib import contextmanager

@contextmanager
def database_connection():
    """
    Create Database connection instance
    """

    DB_USER = "postgres"
    DB_PASSWORD = "12345"
    DB_HOST = "127.0.0.1"
    DB_PORT = "5432"
    DB_NAME = "orphansdb"
    
    connection = None  
    cursor = None
    
    try:
        connection = psycopg2.connect(user=DB_USER, password=DB_PASSWORD,
                                      host=DB_HOST, port=DB_PORT, database=DB_NAME)
        cursor = connection.cursor()
        yield cursor
    except (Exception, psycopg2.Error) as e:
        print(f"Error in database connection: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def check_script_run_status(source):
    """
    Check the database on what is the script running status
    """
    with database_connection() as cursor:
        query = f"""
            SELECT script.exec_status
            from mlpipeline_scriptexecutions script
            join mlpipeline_camera cam on script.exec_camera_id=cam.id
            where camera_ip = '{source}'
            order by script.exec_start_time desc limit 1;
        """
        cursor.execute(query)
        sql_result = cursor.fetchone()
    return sql_result[0]


def save_people_database(people_stats, source):
    TABLE_NAME = "mlpipeline_detection"
    with database_connection() as cursor:
        count = 0
        for person in people_stats.values():
            query = f"""
            INSERT INTO mlpipeline_detection (mood_name, activity_name, recorded_date, camera_id, profile_id)
            SELECT
                %s AS mood_name,
                %s AS activity_name,
                %s AS recorded_date,
                c.id AS camera_id,
                p.id AS profile_id
            FROM
                mlpipeline_camera c
            JOIN
                profile p ON p.profile_name = %s
            WHERE
                c.camera_ip = %s;
            """
            count += 1
            cursor.execute(query, (person.mood, person.action, time, person.name[:-1], source))
        print(count, f"Records inserted successfully into {TABLE_NAME} table")

def insert_prediction(people_stats):
    """
    Insert predictions into a PostgreSQL database table.

    This function takes a dictionary of people's statistics (e.g., name, mood, action) and inserts these
    statistics into a specified PostgreSQL database table. It establishes a connection to the database,
    iterates through the provided people's statistics, and inserts each person's data into the table.

    Args:
        people_stats (dict): A dictionary containing objects of people with attributes like name, mood, and action.

    Returns:
        None.
    """
    # Establish a connection to the PostgreSQL database
    try:
        connection = psycopg2.connect(user=DB_USER,
                                      password=DB_PASSWORD,
                                      host=DB_HOST,
                                      port=DB_PORT,
                                      database=DB_NAME)
        cursor = connection.cursor()

        time = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        # SQL query to insert the prediction into the specified table
        postgres_insert_query = f"""INSERT INTO {TABLE_NAME}(profile_name, mood_name, activity_name,
                                                             recorded_date, camera_id, profile_id, profile_role)
                                     VALUES (%s,%s, %s, %s, %s, %s, %s)"""


        count = 0
        for person in people_stats.values():
            record_to_insert = (person.name, person.mood, person.action, time, 1, 1, person.name[:-1])
            cursor.execute(postgres_insert_query, record_to_insert)
            count += 1

        # Commit the changes to the database after all records are inserted
        connection.commit()
        print(count, f"Records inserted successfully into {TABLE_NAME} table")

    except psycopg2.Error as error:
        print("Error inserting records:", error)

    finally:
        # Close the cursor and connection in the finally block
        if cursor:
            cursor.close()
        if connection:
            connection.close()



def select_all_records(table_name):
    """
    Retrieve all records from a specified PostgreSQL database table.

    This function establishes a connection to the PostgreSQL database using the provided credentials,
    executes an SQL query to select all rows from the specified table, and prints the retrieved records.

    Args:
        table_name (str): The name of the PostgreSQL database table to retrieve records from.

    Returns:
        None: This function retrieves and prints records but does not return any values.
    """
    try:
        # Establish a connection to the PostgreSQL database
        connection = psycopg2.connect(user=DB_USER,
                                      password=DB_PASSWORD,
                                      host=DB_HOST,
                                      port=DB_PORT,
                                      database=DB_NAME)
        cursor = connection.cursor()

        # SQL command to select all rows from the specified table
        postgres_select_query = f"SELECT * FROM {table_name};"
        cursor.execute(postgres_select_query)

        # Fetch all the rows and print them
        records = cursor.fetchall()
        print(records)

        for row in records:
            print(row)

    except (Exception, psycopg2.Error) as error:
        # Handle any exceptions that may occur during the operation
        print(f"Error selecting records from {table_name} table:", error)

    finally:
        # Close the database connection, regardless of success or failure
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

def insert_into_camera(id, camera_name, camera_ip, room_details, connected):
    """
    Insert camera information into the PostgreSQL database.

    Args:
        id (int): Unique identifier for the camera.
        camera_name (str): Name of the camera.
        camera_ip (str): IP address of the camera.
        room_details (str): Details about the room where the camera is located.
        connected (bool): Boolean indicating whether the camera is connected.

    Returns:
        None: This function does not return any values.

    Raises:
        psycopg2.Error: If there's an error during the database operation.

    """
    try:
        # Establish a connection to the PostgreSQL database
        connection = psycopg2.connect(user=DB_USER,
                                      password=DB_PASSWORD,
                                      host=DB_HOST,
                                      port=DB_PORT,
                                      database=DB_NAME)

        cursor = connection.cursor()  # Create a cursor object for executing SQL queries

        # Get the current date and time
        current_datetime = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')

        # SQL code to insert data into the 'mlpipeline_camera' table
        sql_query = "INSERT INTO mlpipeline_camera (id, camera_name, camera_ip, room_details, connected, created_date, updated_date) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        data_to_insert = (id, camera_name, camera_ip, room_details, connected, current_datetime, current_datetime)

        # Execute the SQL query to insert data
        cursor.execute(sql_query, data_to_insert)

        # Commit the changes to the database
        connection.commit()

        # Close the cursor and the database connection
        cursor.close()
        connection.close()

    except psycopg2.Error as error:
        # Handle any database-related errors
        print(f"Error inserting data into the database: {error}")


if __name__ == "__main__":
    select_all_records(TABLE_NAME)

    # select_all_records("mlpipeline_camera")
