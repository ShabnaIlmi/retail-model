import mysql.connector
from mysql.connector import Error

# Database connection details
db_config = {
    "host": "localhost",          # MySQL host
    "user": "your_username",      # MySQL username
    "password": "your_password",  # MySQL password
    "database": "retail_db"        # Database name
}

# Function to create and return a connection
def create_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        if conn.is_connected():
            print("✅ Connected to MySQL database")
            return conn
        else:
            print("❌ Failed to connect to MySQL")
            return None
    except Error as e:
        print(f"❌ Error while connecting to MySQL: {e}")
        return None

# Function to insert detection data into MySQL (with image blob)
def insert_detection_data(unique_id, item_name, quantity, image_blob):
    conn = None
    cursor = None
    try:
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            query = """
                INSERT INTO detection_data (id, item_name, quantity, image)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (unique_id, item_name, quantity, image_blob))
            conn.commit()
            print(f"✅ Data inserted successfully for item: {item_name} (ID: {unique_id})")
    except Error as e:
        print(f"❌ Error inserting data: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

# Function to insert detection data into MySQL (with image path instead of blob) - optional
def insert_detection_data_with_path(unique_id, item_name, quantity, image_path):
    conn = None
    cursor = None
    try:
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            query = """
                INSERT INTO detection_data (id, item_name, quantity, image_path)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (unique_id, item_name, quantity, image_path))
            conn.commit()
            print(f"✅ Data inserted successfully (path) for item: {item_name} (ID: {unique_id})")
    except Error as e:
        print(f"❌ Error inserting data with path: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
