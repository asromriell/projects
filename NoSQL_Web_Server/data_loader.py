import json
import psycopg2, psycopg2.extras
import csv


# this is the set of functions I used to build my database
# the source of my data is Amazon Reviews:  CD's and Vinyl
# I subsetted the data with Rolling Stone Magazine's Top 100 Artists
# I only used the top 10 artists from this list.
# http://www.rollingstone.com/music/lists/100-greatest-artists-of-all-time-19691231

# i had to transform it first because i was only interested in certain categories.
# i was not interested in the full text reviews or descriptions.
# this significantly reduced the memory load
# here are also functions for dropping and creating the db


# DSN location of the AWS - RDS instance
DB_DSN = "host=*** " \
         "dbname=*** " \
         "user=*** " \
         "password=***"

# DB_DSN = "host='localhost' dbname='project'"

# location of the input data file
INPUT_DATA1 = '/Users/alexromriell/Documents/FallModule2/NoSQL/meta_CDs_and_Vinyl_FIXED.json'
INPUT_DATA2 = '/Users/alexromriell/Documents/FallModule2/NoSQL/reviews_CDs_and_Vinyl.json'
# INPUT_DATA1 = '/Users/alexromriell/Documents/FallModule2/NoSQL/meta_first50_fixed.json'
# INPUT_DATA2 = '/Users/alexromriell/Documents/FallModule2/NoSQL/reviews_first100.json'


def transform_data(filename1, filename2):
    """
    :param filename: the filename of the data that will be transformed
    :return: list of tuples to be inserted into the db
    """
    data1 = list()
    data2 = list()
    with open(filename1, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            try:
                asin = str(tmp['asin'])
            except KeyError:
                asin = None
            try:
                title = str(tmp['title'])
            except KeyError:
                title = None
            try:
                imUrl = str(tmp['imUrl'])
            except KeyError:
                imUrl = None
            try:
                salesRank = tmp['salesRank']
            except KeyError:
                salesRank = None
            try:
                categories = tmp['categories']
            except KeyError:
                categories = None
            try:
                price = tmp['price']
            except KeyError:
                price = None

            tup1 = (asin, title, price, json.dumps(salesRank), imUrl, json.dumps(categories))
            data1.append(tup1)

    with open(filename2, 'r') as w:
        for item in w:
            tmp = json.loads(item)
            try:
                asin = tmp['asin']
            except KeyError:
                asin = None
            try:
                summary = tmp['summary']
            except KeyError:
                summary = None
            try:
                helpful = tmp['helpful']
            except KeyError:
                helpful = None
            try:
                unixReviewTime = tmp['unixReviewTime']
            except KeyError:
                unixReviewTime = None
            try:
                reviewTime = tmp['reviewTime']
            except KeyError:
                reviewTime = None
            try:
                overall = tmp['overall']
            except KeyError:
                overall = None

            tup2 = (asin, json.dumps(helpful), summary, overall, reviewTime, unixReviewTime)
            data2.append(tup2)

    with open('tup1.csv', 'wb') as myfile1:
        wr = csv.writer(myfile1, delimiter=',')
        for row in data1:
            wr.writerow(row)

    with open('tup2.csv', 'wb') as myfile2:
        wr = csv.writer(myfile2, delimiter=',')
        for row in data2:
            wr.writerow(row)

    return


def drop_table():
    """
    drops the table 'meta', 'review', 'top10', if it exists
    :return:
    """
    sql1 = "DROP TABLE IF EXISTS meta;"
    conn1 = psycopg2.connect(dsn=DB_DSN)
    cur1 = conn1.cursor()
    cur1.execute(sql1)
    conn1.commit()

    sql2 = "DROP TABLE IF EXISTS reviews;"
    conn2 = psycopg2.connect(dsn=DB_DSN)
    cur2 = conn2.cursor()
    cur2.execute(sql2)
    conn2.commit()

    sql3 = "DROP TABLE IF EXISTS top10;"
    conn3 = psycopg2.connect(dsn=DB_DSN)
    cur3 = conn3.cursor()
    cur3.execute(sql3)
    conn3.commit()

def create_table():
    """
    creates postgres tables with columns ...
    :return:
    """
    sql1 = "CREATE TABLE meta (" \
          "asin TEXT, " \
          "title TEXT, " \
          "price NUMERIC , " \
          "salesRank JSON, " \
          "imURL TEXT, " \
          "categories JSON);"
    conn1 = psycopg2.connect(dsn=DB_DSN)
    cur1 = conn1.cursor()
    cur1.execute(sql1)
    conn1.commit()

    sql2 = "CREATE TABLE reviews (" \
          "asin TEXT, " \
          "helpful JSON, " \
          "summary TEXT, " \
          "overall NUMERIC, " \
          "reviewTime TEXT, " \
          "unixReviewTime INT);"
    conn2 = psycopg2.connect(dsn=DB_DSN)
    cur2 = conn2.cursor()
    cur2.execute(sql2)
    conn2.commit()


def insert_data():
    """
    inserts the data using copy_expert
    :param data: a list of tuples with order ...
    :return:
    """

    try:
        sql1 = "COPY meta FROM stdin DELIMITER ',' CSV; "
        f1 = open('tup1.csv', 'r')
        conn1 = psycopg2.connect(dsn=DB_DSN)
        cur1 = conn1.cursor()
        cur1.copy_expert(sql1, f1)
        conn1.commit()
    except psycopg2.Error as e:
        print e.message
    else:
        cur1.close()
        conn1.close()

    try:
        sql2 = "COPY reviews FROM stdin DELIMITER ',' CSV; "
        f2 = open('tup2.csv', 'r')
        conn2 = psycopg2.connect(dsn=DB_DSN)
        cur2 = conn2.cursor()
        cur2.copy_expert(sql2, f2)
        conn2.commit()
    except psycopg2.Error as e:
        print e.message
    else:
        cur2.close()
        conn2.close()


def create_top10():
    """
    creates a new table, top10, from the two tables already created: meta and reviews

    :return: nothing; creates new table with values
    """

    sql = """
    CREATE TABLE top10 as
        select iq.title, iq.asin, iq.price,
        case when iq.title LIKE '%The Beatles%' then 'beatles'
        when iq.title LIKE '%Bob Dylan%' then 'bob dylan'
        when iq.title LIKE '%Elvis Presley%' then 'elvis'
        when iq.title LIKE '%The Rolling Stones%' then 'rolling stones'
        when iq.title LIKE '%Chuck Berry%' then 'chuck berry'
        when iq.title LIKE '%Jimi Hendrix%' then 'hendrix'
        when iq.title LIKE '%James Brown%' then 'james brown'
        when iq.title LIKE '%Little Richard%' then 'little richard'
        when iq.title LIKE '%Aretha Franklin%' then 'aretha franklin'
        when iq.title LIKE '%Ray Charles%' then 'ray charles'
        END as artist
        from
        (select title, asin, price
        from meta
        where title LIKE '%The Beatles%'
             OR title LIKE '%Bob Dylan%'
             OR title LIKE '%Elvis Presley%'
             OR title LIKE '%The Rolling Stones%'
             OR title LIKE '%Chuck Berry%'
             OR title LIKE '%Jimi Hendrix%'
             OR title LIKE '%James Brown%'
             OR title LIKE '%Little Richard%'
             OR title LIKE '%Aretha Franklin%'
             OR title LIKE '%Ray Charles%') as iq
		 """
    conn = psycopg2.connect(dsn=DB_DSN)
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    return


if __name__ == '__main__':

    import time
    start_time = time.time()
    # running this program as a main file will perform ALL the ETL
    # it will extract and transform the data from it file

    print "transforming data"
    # data = transform_data(INPUT_DATA1, INPUT_DATA2)
    transform_data(INPUT_DATA1, INPUT_DATA2)

    # drop the db
    print "dropping table"
    drop_table()

    # create the db
    print "creating table"
    create_table()

    # insert the data
    print "inserting data"
    insert_data()

    # create top10 table
    print "creating top10 table from inserted data"
    create_top10()

    print "My program took", time.time() - start_time, "seconds to run"
