import sqlite3

connection = sqlite3.connect('learning_to_plan.db')

# remove 10 tasks from the database

cursor = connection.cursor()
cursor.execute("SELECT * FROM task")
results = cursor.fetchall()
print(f"Number of tasks before deletion: {len(results)}")
cursor.execute("DELETE FROM task WHERE id IN (SELECT id FROM task ORDER BY id LIMIT 10)")
cursor.execute("SELECT * FROM task")
results = cursor.fetchall()
print(f"Number of tasks after deletion: {len(results)}")
connection.commit()
cursor.close()
connection.close()