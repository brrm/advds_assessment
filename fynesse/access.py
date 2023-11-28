from .config import *
from .utils import *

import pandas as pd
import csv
import pymysql
from typing import Optional

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

UK_PP_DATA_RANGE = (1995, 2022)

class DBConn:
    def __init__(self):
        """
        Create a database connection to the MariaDB database
        as specified in the config.
        """
        user = config['db_username']
        password = config['db_password']
        host = config['db_url']
        port = config['db_port']
        database = config['db_name']

        self.conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )

    def select_top(self, table: str,  n: int):
        """
        Query n first rows of the table
        :param conn: the Connection object
        :param table: The table to query
        :param n: Number of rows to query
        """
        cur = self.conn.cursor()
        cur.execute(f'SELECT * FROM {table} LIMIT {n}')
    
        rows = cur.fetchall()
        return rows

    def select_where(self, table: str, cols: str, condition: str = None):
        cur = self.conn.cursor()
        q = f'SELECT {cols} FROM {table}{"" if condition is None else f" WHERE {condition}"}'
        print(q)
        cur.execute(q)
    
        rows = cur.fetchall()
        return rows

    def select_aggregate(self, table: str, cols: str, groupby: str):
        cur = self.conn.cursor()
        cur.execute(f'SELECT {cols} FROM {table} GROUP BY {groupby}')
    
        rows = cur.fetchall()
        return rows

    def get_pc_bbox(self, north, south, east, west, min_year: Optional[int] = None, max_year: Optional[int] = None):
        if not(min_year is None) and min_year <= UK_PP_DATA_RANGE[0]:
            min_year = None
        if not(max_year is None) and max_year >= UK_PP_DATA_RANGE[1]:
            max_year = None
        
        q = f"""SELECT * FROM `prices_coordinates_data`
                WHERE longitude >= {west} AND longitude <= {east} AND latitude >= {south} AND latitude <= {north}
                {f' AND year_of_transfer >= {min_year}' if not(min_year is None) else ''}
                {f' AND year_of_transfer <= {max_year}' if not(max_year is None) else ''};"""
        
        return pd.read_sql(q, self.conn)

    # --- Below methods are for uploading/setting up the DB --- 
    def upload_pp_data(self, year: int, 
                       pp_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/",
                       cleanup = True):
            filename = f'pp-{year}.csv'
        
            download_file(pp_url+filename, filename)
            self.upload_csv(filename, 'pp_data')
            
            if cleanup:
                os.remove(filename)
            
            print(f'Succesfully uploaded {year} PP data')

    def upload_csv(self, filename: str, table: str):
        with self.conn.cursor() as cur:
            cur.execute(f"""LOAD DATA LOCAL INFILE '{filename}' INTO TABLE {table} 
                            FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '"' 
                            LINES STARTING BY '' TERMINATED BY '\n';""")
        self.conn.commit()
    
    def join_prices_coordinates(self, year: int):
        # Join tables and write to a CSV
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT price as price, date_of_transfer, year_of_transfer, pp.postcode as postcode, property_type,
                new_build_flag, tenure_type, locality, town_city, district, county, 
                country, latitude, longitude, pp.db_id as db_id
                FROM `pp_data` as pp join `postcode_data` as pc 
                ON pp.postcode = pc.postcode
                WHERE year_of_transfer = {year};""")
    
            results = cur.fetchall()
    
            filename = f'price_coordinates_{year}.csv'
    
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
    
                for row in results:
                    writer.writerow(row)

        # Upload the CSV and cleanup
        self.upload_csv(filename=filename, table='prices_coordinates_data')
        os.remove(filename)
    
        print(f'Successfully joined {year} PP and postcode data')


def download_file(url: str, filename: str):
    """
    Download a file from a URL and save it to a local file.
    
    :param url: URL of the file to download
    :param filename: Name of the file to save locally
    """
    response = requests.get(url)

    # Check for successful response before writing
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        raise f"Failed to download. Status code: {response.status_code}"


