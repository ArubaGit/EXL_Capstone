import mysql.connector

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="!@Qwerty25",  
        database="churn_db"
    )
import csv

def insert_data_from_csv(file_path):
    conn = connect_db()
    cursor = conn.cursor()

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0  # initialize a counter
        for row in reader:
            if count >= 20:
                break  # stop after 20 rows

            query = """
                INSERT INTO customers 
                (CustomerID, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Churn)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                row['CustomerID'],
                row['Gender'] if row['Gender'] else None,
                int(row['Age']) if row['Age'] else None,
                int(row['Tenure']) if row['Tenure'] else None,
                float(row['Balance']) if row['Balance'] else 0.0,
                int(row['NumOfProducts']) if row['NumOfProducts'] else None,
                1 if row['HasCrCard'].strip().lower() == 'yes' else 0,
                1 if row['IsActiveMember'].strip().lower() == 'yes' else 0,
                float(row['EstimatedSalary']) if row['EstimatedSalary'] else 0.0,
                1 if row['Churn'].strip().lower() in ['yes', 'maybe'] else 0
            )
            cursor.execute(query, values)
            count += 1  # increment the counter

    conn.commit()
    cursor.close()
    conn.close()

def display_all_customers():
    conn = connect_db()
    cursor = conn.cursor()

    # cursor.execute("SELECT * FROM customers")
    # results = cursor.fetchall()

    # for row in results:
    #     print(row)

    query = "SELECT * FROM customers LIMIT 20"
    cursor.execute(query)
    results = cursor.fetchall()
    
    # Print column headers
    column_names = [i[0] for i in cursor.description]
    print("\t".join(column_names))
    
    # Print each row
    print("______________________________DB DATA________________________________")
    for row in results:
        print("\t".join(str(item) for item in row))
    
    cursor.close()
    conn.close()
