import csv
import pandas as pd
from geopy.geocoders import Nominatim

df = pd.read_excel('Areas y población Mun_COL 2005_2017.xlsx').iloc[:-1,:]

write_file = "output.csv"

geolocator = Nominatim()
country ="Colombia"

with open(write_file, "w") as output:
    for place in df['MPIO'].values:
        try:
            city = place
            loc = geolocator.geocode(city+','+ country)
            coordinates = city+', '+str(loc.latitude)+', '+str(loc.longitude)
            output.write(coordinates + '\n')
            print(coordinates)
            
            #if city == 'Amagá':
            #    break
        
        except:
            output.write(city + '\n')
            print(city)
