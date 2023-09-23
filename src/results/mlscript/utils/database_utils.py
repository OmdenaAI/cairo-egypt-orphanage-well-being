import psycopg2
import datetime
from contextlib import contextmanager

DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "127.0.0.1"
DB_PORT = "5432"
DB_NAME = "orphan"

@contextmanager
def database_connection():
    """
    Create Database connection instance
    """

    DB_USER = "postgres"
    DB_PASSWORD = "1234"
    DB_HOST = "127.0.0.1"
    DB_PORT = "5432"
    DB_NAME = "orphan"

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




def insert_into_database(people_stats):
    """
    Insert camera information into the PostgreSQL database.

    Args:
        people_stats: dict of people statistics

    Returns:
        None: This function does not return any values.

    Raises:
        psycopg2.Error: If there's an error during the database operation.

    """
    try:
        # Establish a connection to the PostgreSQL database
        connection = psycopg2.connect(user="postgres",
                                      password="1234",
                                      host="127.0.0.1",
                                      port="5432",
                                      database="orphan")

        cursor = connection.cursor()  # Create a cursor object for executing SQL queries

        # Get the current date and time
        current_datetime = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')

        # SQL code to insert data into the 'mlpipeline_camera' table
        sql_query = "INSERT INTO mlpipeline_detection (recorded_date, camera_id, profile_id, mood_name, activity_name) VALUES (%s, %s, %s, %s, %s)"

        for person in people_stats.values():
            data_to_insert = (current_datetime, 40, 2, person.mood, person.action)

            # Execute the SQL query to insert data
            cursor.execute(sql_query, data_to_insert)

        # Commit the changes to the database after the loop
        connection.commit()

        # Close the cursor and the database connection
        cursor.close()
        connection.close()

    except psycopg2.Error as error:
        # Handle any database-related errors
        print(f"Error inserting data into the database: {error}")


def check_script_run_status(source):
    """
    Check the database on what is the script running status
    """
    with database_connection() as cursor:
        query = f"""
            SELECT script.exec_status
            from mlpipeline_scriptexecutions script
            join mlpipeline_camera cam on script.exec_camera_id=cam.id
            where camera_ip = '{str(source)}'
            order by script.exec_start_time desc limit 1;
        """
        cursor.execute(query)
        sql_result = cursor.fetchone()
    return sql_result[0]



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
    DB_USER = "postgres"
    DB_PASSWORD = "1234"
    DB_HOST = "127.0.0.1"
    DB_PORT = "5432"
    DB_NAME = "orphan"
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

def insert_into_camera(camera_ip, room_details, connected):
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
    DB_USER = "postgres"
    DB_PASSWORD = "1234"
    DB_HOST = "127.0.0.1"
    DB_PORT = "5432"
    DB_NAME = "orphan"
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
        sql_query = "INSERT INTO mlpipeline_camera (camera_ip, room_details, connected, created_date, updated_date) VALUES (%s, %s, %s, %s, %s)"
        data_to_insert = (camera_ip, room_details, connected, current_datetime, current_datetime)

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
    # select_all_records(TABLE_NAME)
    # insert()
    select_all_records("mlpipeline_detection")
    # select_all_records("mlpipeline_camera")

    # time = datetime.datetime.today()
    # select_all_records("mlpipeline_detection")
    # insert_into_camera("0", "Room B", True)
    # select_all_records("mlpipeline_camera")
