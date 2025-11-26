import sqlite3

def init_db():
    try:
        with sqlite3.connect('SantaDB.db') as conn:
            with open('schema.sql', 'r') as f:
                conn.executescript(f.read())
        print("Database SantaDB.db has been initialized successfully.")
    except sqlite3.Error as e:
        print(f"An error occurred while initializing the database: {e}")
    except FileNotFoundError as e:
        print(f"Could not find file 'schema.sql': {e}")

if __name__ == '__main__':
    init_db()