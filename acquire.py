import pandas as pd
import os
from env import host, user, password

def get_titanic_data():
    filename = "titanic.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        titanic_sql = """
        SELECT * 
        FROM passengers
        ;
        """

        titanic_url = f'mysql+pymysql://{user}:{password}@{host}/titanic_db'

        return pd.read_sql(titanic_sql, titanic_url)
    
        
def get_iris_data():
    filename = "iris.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        iris_sql = """
        SELECT * 
        FROM measurements
        JOIN species USING (species_id)
        ;
        """

        iris_url = f'mysql+pymysql://{user}:{password}@{host}/iris_db'

        return pd.read_sql(iris_sql, iris_url)

def get_telco_data():
    filename = "telco.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        telco_sql = """
        SELECT * 
        FROM customers
        JOIN contract_types USING (contract_type_id)
        JOIN payment_types USING (payment_type_id)
        JOIN internet_service_types USING (internet_service_type_id)
        ;
        """

        telco_url = f'mysql+pymysql://{user}:{password}@{host}/telco_churn'

        return pd.read_sql(telco_sql, telco_url)

