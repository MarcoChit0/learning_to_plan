import sqlite3

connection = sqlite3.connect('generated_plan.db')
cursor = connection.cursor()

query = "SELECT * FROM generated_plan"
cursor.execute(query)
for r in cursor.fetchall():
    print(r)

connection.close()