import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('logs/logs.db')
cursor = conn.cursor()

# Function to view all logs
def view_logs():
    cursor.execute("SELECT * FROM logs")
    rows = cursor.fetchall()
    if rows:
        print("Current logs:")
        for row in rows:
            print(row)
    else:
        print("No logs found.")

# Function to delete all logs
def delete_all_logs():
    cursor.execute("DELETE FROM logs")
    conn.commit()
    print("All logs have been deleted.")

# Example usage:
view_logs()          # View logs before deletion
#delete_all_logs()  # Uncomment to delete all logs
# view_logs()        # View logs after deletion to confirm

conn.close()
