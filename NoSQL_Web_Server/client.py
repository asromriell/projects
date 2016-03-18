import httplib

# use this server for dev
# SERVER = 'localhost:5000'

# use this server for prod, once it's on ec2
SERVER = '***'


def get_product_totals_by_artist():
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/artists/product_totals')
    resp = h.getresponse()
    out = resp.read()
    return out


def get_summaries_of_products():
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/products/summaries')
    resp = h.getresponse()
    out = resp.read()
    return out


def get_summaries_of_products_artist(artist):
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/products/summaries/'+artist)
    resp = h.getresponse()
    out = resp.read()
    return out


def get_artists_rankings():
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/artists/ranking')
    resp = h.getresponse()
    out = resp.read()
    return out


def get_artists_rankings_by_year(year):
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/artists/ranking/'+year)
    resp = h.getresponse()
    out = resp.read()
    return out


if __name__ == '__main__':
    print "************************************************"
    print "test of my flask app running at ", SERVER
    print "created by Alex Romriell"
    print "************************************************"
    print " "
    print "******** counts artist products **********"
    print get_product_totals_by_artist()
    print " "
    print "******** get summaries of products **********"
    print get_summaries_of_products()
    print " "
    print "******** get summaries of products - beatles **********"
    print get_summaries_of_products_artist('beatles')
    print " "
    print "******** get artist rankings **********"
    print get_artists_rankings()
    print " "
    print "******** get artist ranking for given year - 2001 **********"
    print get_artists_rankings_by_year('2001')
