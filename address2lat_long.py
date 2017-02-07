from geopy.geocoders import Nominatim
import pandas as pd
from time import sleep


def convert_address(street_add):
    geolocator = Nominatim()
    try:
        location = geolocator.geocode(street_add)
        # print(location.address)
        lat, long = location.latitude, location.longitude
        # print lat, long
    except AttributeError as e:
        print e
        print 'Could not find coords for', street_add
        lat, long = 33.873898, -118.383984
    return lat, long


if __name__ == '__main__':
    # path = 'C:\Users\J60087\Documents\\fun_projects\strawberries\orders.csv'
    # data = pd.read_csv(path)
    #
    # lat_long_list = list()
    # cnt = 0
    # for location in data['Address']:
    #     location = location.strip('\n')
    #     print location
    #     lat, long = convert_address(location)
    #     sleep(2)
    #     lat_long_list.append([lat, long, location])
    #
    #     # cnt += 1
    #     # if cnt > 1:
    #     #     break
    # print ''
    # print lat_long_list
    # with open('lat_long_list.csv', 'w') as f:
    #     for item in lat_long_list:
    #         string = str(item[0]) + ',' + str(item[1]) + ',' + str(item[2]) + '\n'
    #         f.write(string)

    lat_long_list = pd.read_csv('C:\Users\J60087\Documents\\fun_projects\strawberries\lat_long_list.csv')
    lat_long_list = lat_long_list.sort_values(by=['lat', 'long'])
    # print lat_long_list['Name']

    colors = [
        'https://asromriell.github.io/images/purple_MarkerA.png',
        'https://asromriell.github.io/images/red_MarkerA.png',
        'https://asromriell.github.io/images/green_MarkerB.png',
        'https://asromriell.github.io/images/blue_MarkerC.png',
        'https://asromriell.github.io/images/orange_MarkerD.png',
        'https://asromriell.github.io/images/MarkerE.png',
        'https://asromriell.github.io/images/paleblue_MarkerF.png'
        ]
    c_cnt = 0
    color_list = list()
    html_address = list()
    for i in range(len(lat_long_list)):
        name = lat_long_list['Name'].iloc[[i]].values[0]
        address = lat_long_list['Address'].iloc[[i]].values[0]
        order_amt = lat_long_list['# of Boxes'].iloc[[i]].values[0]
        amt_due = lat_long_list['Amount Due'].iloc[[i]].values[0]
        amt_paid = lat_long_list['Paid'].iloc[[i]].values[0]

        if i % 8 == 0:
            c_cnt += 1

        color_list += [colors[c_cnt]]

        print name, address, order_amt, amt_due, amt_paid

        html_address += ["""
              '<div id="content">'+
              '<div id="siteNotice">'+
              '</div>'+
              '<h1 id="firstHeading" class="firstHeading">{0}</h1>'+
              '<div id="bodyContent">'+
              '<p><b>Address: </b>{1}</p>'+
              '<p><b>Order: </b>{2}</p>'+
              '<p><b>Amount Due: </b>{3}</p>'+
              '<p><b>Paid: </b>{4}</p>'+
              '</div>'+
              '</div>'
            """.format(name, address, order_amt, amt_due, amt_paid)]

    lat_long_list['color'] = color_list
    lat_long_list['html_address'] = html_address

    lat_long_list.to_csv('lat_long2.csv', index=False)

    data = pd.read_csv('C:\Users\J60087\Documents\\fun_projects\strawberries\lat_long2.csv')
    for i in range(len(data)):
        print i, data['lat'].iloc[[i]].values[0]

    with open('lat_long_html.txt', 'w') as f:
        for i in range(len(data)):
            line = '[' + str(data['lat'].iloc[[i]].values[0]) + ',' + \
                   str(data['long'].iloc[[i]].values[0]) + ', ' + \
                   str(data['html_address'].iloc[[i]].values[0]) + ', "' + \
                   str(data['color'].iloc[[i]].values[0]) + \
                   '"],\n'
            f.write(line)

