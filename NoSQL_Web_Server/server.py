from flask import Flask, request, jsonify, render_template
import psycopg2, psycopg2.extras

# DSN location of the AWS - RDS instance
DB_DSN = "host=*** " \
         "dbname=*** " \
         "user=*** " \
         "password=***"
# DB_DSN = 'host=localhost dbname=project'
app = Flask(__name__)


@app.route('/')
def default():

    return render_template('welcome_page.html')


@app.route('/artists/product_totals')
def get_product_totals_by_artist():
    """Queries the sum total of products for each artist.

    Returns
    -------
    JSON object
    """
    # init an output
    out = dict()
    try:
        sql = "select artist, sum(price) as total, count(price) " \
              "from top10 " \
              "group by artist;"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        rs = cur.fetchall()
        for item in rs:
            artist, total, count = item
            out[artist] = {"num_products": str(count), "sum_price_all_products": str(total)}

    except psycopg2.Error as e:
        print e.message
    else:
        cur.close()
        conn.close()

    return jsonify(out)


@app.route('/products/summaries')
def get_summaries_of_products():
    """
    Queries the summary, overall rating, price, and helpful review count for each product

    Returns
    -------
    JSON object with the above categories by product
    """
    # init an output
    out = dict()
    try:
        sql = "select price, title, artist, summary, overall, helpful " \
              "from " \
              "(select asin, title, artist, price " \
              "from top10) as LHS " \
              "INNER JOIN " \
              "(select asin, summary, overall, helpful " \
              "from reviews) as RHS " \
              "ON LHS.asin = RHS.asin"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        rs = cur.fetchall()
        for item in rs:
            price, title, artist, summary, overall, helpful = item
            out[title] = {"artist": artist, "price": str(price), "summary": summary, "overall": str(overall), "helpful": helpful}

    except psycopg2.Error as e:
        print e.message
    else:
        cur.close()
        conn.close()

    return jsonify(out)


@app.route('/products/summaries/<artist>')
def get_summaries_of_products_artist(artist):
    """
    Queries the summary, overall rating, price, and helpful review count for each product, by specified artist

    Parameters
    ----------
    artist : string
        A string indicating which artist the user wants to view product summaries of
    Returns
    -------
    JSON object containing the above categories
    """
    # init an output
    out = dict()
    try:
        sql = "select price, title, artist, summary, overall, helpful " \
              "from" \
              "(select asin, title, artist, price " \
              "from top10) as LHS " \
              "INNER JOIN " \
              "(select asin, summary, overall, helpful " \
              "from reviews) as RHS " \
              "ON LHS.asin = RHS.asin " \
              "WHERE artist = %s"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql, (artist,))
        rs = cur.fetchall()
        for item in rs:
            price, title, artist, summary, overall, helpful = item
            out[title] = {"artist": artist, "price": str(price), "summary": summary,
                          "overall": str(overall), "helpful": helpful}

    except psycopg2.Error as e:
        print e.message
    else:
        cur.close()
        conn.close()

    return jsonify(out)


@app.route('/artists/ranking')
def get_artists_rankings():
    """
    Queries for the total reviews for each artist by year

    Returns
    -------
    JSON object with count, artist, and year
    """
    # init an output
    out = dict()
    try:
        sql = "select count(artist), year, artist " \
              "from " \
              "(select LHS.asin, year, artist, title, summary " \
              "from " \
              "(select asin, date_part('year',to_timestamp(unixreviewtime)) as year, summary from reviews) as LHS " \
              "INNER JOIN (select asin, artist, title from top10) as RHS  " \
              "ON LHS.asin = RHS.asin) as inner_q " \
              "group by year, artist " \
              "order by year DESC, count DESC"

        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        rs = cur.fetchall()
        for item in rs:
            count, year, artist = item
            out[count] = {"year": year, "artist": artist}

    except psycopg2.Error as e:
        print e.message
    else:
        cur.close()
        conn.close()

    return jsonify(out)


@app.route('/artists/ranking/<year>')
def get_artists_rankings_by_year(year):
    """
    Queries for the total reviews for each artist for a given year

    Parameters
    ----------
    year : string
        The year of interest, between 1998 - 2014
    Returns
    -------
    JSON object containing counts of reviews for each artist for a given year
    """
    # init an output
    out = dict()
    try:
        sql = "select count(artist), year, artist " \
              "from " \
              "(select LHS.asin, year, artist, title, summary " \
              "from " \
              "(select asin, date_part('year',to_timestamp(unixreviewtime)) as year, summary from reviews) as LHS " \
              "INNER JOIN (select asin, artist, title from top10) as RHS  " \
              "ON LHS.asin = RHS.asin) as inner_q " \
              "where year = %s" \
              "group by year, artist " \
              "order by year DESC, count DESC"

        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql, (year,))
        rs = cur.fetchall()
        for item in rs:
            count, year, artist = item
            out[count] = {"year": year, "artist": artist}

    except psycopg2.Error as e:
        print e.message
    else:
        cur.close()
        conn.close()

    return jsonify(out)


if __name__ == "__main__":
    # app.debug = True # only have this on for debugging!
    app.run(host='0.0.0.0') # need this to access from the outside world!
