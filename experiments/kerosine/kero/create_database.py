import sqlite3

def setup_dummy_db():
    conn = sqlite3.connect('local.db')
    cursor = conn.cursor()
    # Create a table with 5 features and 1 label
    cursor.execute('''CREATE TABLE IF NOT EXISTS user_data 
                     (c1 REAL, c2 REAL, c3 REAL, c4 REAL, c5 REAL, label REAL)''')
    
    # Insert some dummy training data
    data = [(i, i*0.5, i+2, i-1, i*1.1, (i*2)+5) for i in range(100)]
    cursor.executemany('INSERT INTO user_data VALUES (?,?,?,?,?,?)', data)
    
    conn.commit()
    conn.close()
    print("Database 'local.db' created successfully.")

setup_dummy_db()
