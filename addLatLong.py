from geopy.geocoders import Nominatim
import pandas as pd
from time import sleep


def convert_address(street_add):

    geolocator = Nominatim()
    try:
        location = geolocator.geocode(street_add)
        # print(location.address)
        lat, lon = location.latitude, location.longitude
        print lat, lon, street_add
    except AttributeError as e:
        # print e
        print 'Could not find coords for', street_add
        lat, lon = None, None
    return lat, lon


def get_lat_lon(df):
    lat_list = list()
    lon_list = list()
    for address in df['Address']:
        lat, lon = convert_address(street_add=address)
        lat_list.append(lat)
        lon_list.append(lon)
        sleep(2)
        # break

    df['lat'] = lat_list
    df['lon'] = lon_list

    return df


def prep_for_js(data, path):
    with open(path, 'w') as f:
        for i in range(len(data)):
            line = '[' + str(data['lat'].iloc[[i]].values[0]) + ',' + \
                    str(data['lon'].iloc[[i]].values[0]) + ',"' + \
                    str(data['Name'].iloc[[i]].values[0]) + '","' + \
                    str(data['marker'].iloc[[i]].values[0]) + \
                    '"],\n'
            f.write(line)

    return


if __name__ == '__main__':
    # path = 'C:\Users\J60087\Documents\\fun_projects\paris\Parisaddresses.csv'
    # data = pd.read_csv(path)
    # new_data = get_lat_lon(df=data)
    # new_data.to_csv('address_with_LatLon.csv')
    data = pd.read_csv('Florida Keys.csv')
    prep_for_js(data, path='florida_keys_js.txt')
