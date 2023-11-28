from .config import *

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from geopandas.tools import sjoin, sjoin_nearest
from typing import Optional

from enum import Enum

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Create visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

class PropertyType(Enum):
    OTHER = 'other'
    FLAT = 'flat'
    TERRACED = 'terraced'
    SEMIDETACHED = 'semidetached'
    DETACHED = 'detached'

    def get_dummies(self):
        return {e.value: (1 if self.value == e.value else 0) for e in PropertyType}


def distance_decay(d, m = 500, pow = 4.0):
    """
    function from distance to closeness
    range is (0,1] with 1 for 0 distance
    :param m: distance at which closeness = 1/e
    :param pow: determines how aggressively closeness decays
    """

    return np.exp(-np.power((d/m), pow))

# TODO: validate this with a heatmap (perhaps side-by-side with prices)
def calculate_closeness(gdf, pois, features: list[str] = None, m = 500, pow = 4.0, max_dist = 1500):
    # gdf should have a geometry column and a unique index
    
    # Lambert projection for distance calculations. Since distances are small, distortions should be negligible
    gdf_points = gdf[['geometry']].to_crs('EPSG:3347')  
    pois = pois.to_crs('EPSG:3347')

    if features is None:
        features = ['school', 'healthcare', 'sustenance', 'entertainment', 'sports', 'public_transport', 
                    'nature', 'water', 'park', 'food_shop', 'commerce', 'tourism']

    closeness_df = pd.DataFrame()
    for f in features:
        # Distance from each point in gdf to the nearest instance of feature f in pois
        dist = gdf_points.sjoin_nearest(pois.loc[pois[f], ['geometry', f]], 
                                        how='left', max_distance=max_dist, distance_col='dist')
        dist = dist[~dist.index.duplicated()] # equidistance results in duplicates so drop them

        closeness_df[f'{f}_closeness'] = distance_decay(dist['dist'].fillna(max_dist).values, m=m, pow=pow)
    
    return closeness_df

def plot_town(ax, gdf, town: str, county: str, buffer=0.01):
    gdf = gdf[(gdf['town_city'] == town) & (gdf['county'] == county)]
    
    west, east = gdf['geometry'].bounds['minx'].min(), gdf['geometry'].bounds['maxx'].max()
    south, north = gdf['geometry'].bounds['miny'].min(), gdf['geometry'].bounds['maxy'].max()
    
    pois = get_pois_labels(north, south, east, west, buffer=buffer)

    # Overpass graph
    graph = ox.graph_from_bbox(north+buffer, south-buffer, east+buffer, west-buffer)
    nodes, edges = ox.graph_to_gdfs(graph)

    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
    
    ax.set_xlim([west-buffer, east+buffer])
    ax.set_ylim([south-buffer, north+buffer])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")    

    # Plot POIs
    pois[pois['park'] | pois['nature']].plot(ax=ax, color="green", alpha=0.2, markersize=5)
    pois[pois['water']].plot(ax=ax, color="blue", alpha=0.2, markersize=5) 
    pois[~(pois['park'] | pois['water'] | pois['nature'])].plot(ax=ax, color="yellow", alpha=0.2, markersize=10)

    # Geopandas' plot() has a bug for log normalisation, so we use scatter() instead
    res = ax.scatter(gdf.geometry.x, gdf.geometry.y, c=gdf['price'], cmap='plasma', alpha=0.8, norm='log', s=4)
    plt.colorbar(res)
    return res
    

def get_pois_labels(north: float, south: float, east: float, west: float, buffer = 0.01, labels = None):
    # Fetch POIs within the given bounding box & relabel them according to given labels
    # Returns GeoFrame with name, geometry cols and a boolean col for each key in labels indicating 
    # whether the POI matches any of the values associated with the key 
    
    if labels is None:
        # Note we don't include buildings because they denote physical shape rather than function
        # e.g. "building=supermarket" means a building has the form of a typical supermarket building 
        # but does not indicate it has an active supermarket sharket
        labels = {
            'school': {'amenity': ['kindergarten', 'school']},
            'healthcare': {'amenity': ['clinic', 'doctors', 'hospital']},
            'sustenance': {'amenity': ['bar', 'cafe', 'fast_food', 'food_court', 'pub', 'restaurant']},
            'entertainment': {'amenity': ['arts_centre', 'casino', 'cinema', 'community_centre', 
                                          'music_venue', 'nightclub', 'theatre', 'library']},
            'sports': {'leisure': ['fitness_centre', 'fitness_station', 'horse_riding', 'ice_rink',
                                  'sports_centre', 'swimming_pool', 'track'],
                      'sport': True},
        
            'public_transport': {'public_transport': ['platform', 'stop_position']},
            'nature': {'natural': ['grassland', 'heath', 'scrub', 'wood']},
            'water': {'natural': ['coastline'],
                      'water': ['river', 'oxbow', 'canal', 'lake', 'reservoir', 'pond', 'lagoon'],
                     },
            'park': {'leisure': ['beach_resort', 'dog_park', 'fishing', 'garden', 'horse_riding', 'nature_reserve',
                                                'park', 'pitch', 'playground', 'swimming_area', 'track', 'golf_course']},
        
            'food_shop': {'shop': ['bakery', 'butcher', 'cheese', 'deli', 'greengrocer', 'food', 'convenience', 'supermarket']},
            'commerce': {'shop': ['department_store', 'mall', 'wholesale', 'bag', 'boutique', 'clothes', 
                                 'jewelry', 'leather', 'shoes', 'tailor', 'watches']},
            'tourism': {'historic': True, 'tourism': True},
        }
    
    # Compact all the tags we need
    tags = dict()
    for label_tags in labels.values():
        for k, v in label_tags.items():
            if isinstance(v, bool):
                tags[k] = v
            else: 
                if k in tags:
                    tags[k] += v.copy()
                else:
                    tags[k] = v.copy()

    pois_raw = ox.features_from_bbox(north+buffer, south-buffer, east+buffer, west-buffer, tags) 
    pois = pois_raw[['geometry', 'name']].copy()
    
    for label, label_tags in labels.items():
        # Conjunction of required tags
        flag = np.full(len(pois), False)
        for k, v in label_tags.items():
            if not(k in pois_raw.columns):
                continue

            if isinstance(v, bool):
                flag = flag | (~pois_raw[k].isna().values)
            else:
                flag = flag | pois_raw[k].isin(v).values

        pois[label] = flag
    
    return pois

def get_uk_counties_gdf():
    # Get GeoDF of UK Ceremonial Counties
    # We use OSMIDs since sometimes there are several matches (e.g. administrative & ceremonial counties)
    uk_counties_osmids = {
        'Bedfordshire': 88082,
        'Berkshire': 88070,
        'Bristol': 5746665,
        'Buckinghamshire': 87460,
        'Cambridgeshire': 87521,
        'Cheshire': 57512,
        'City of London': 51800,
        'Cornwall': 57537,
        'Cumbria': 88065,
        'Derbyshire': 88077,
        'Devon': 57538,
        'Dorset': 2698375,
        'County Durham': 156050,
        'East Riding of Yorkshire': 3125734,
        'East Sussex': 2126747,
        'Essex': 62162,
        'Gloucestershire': 2700308,
        'Greater London': 175342,
        'Greater Manchester': 88084,
        'Hampshire': 2698314,
        'Herefordshire': 10187,
        'Hertfordshire': 57032,
        'Isle of Wight': 154350,
        'Kent': 88071,
        'Lancashire': 118082,
        'Leicestershire': 78309,
        'Lincolnshire': 1916530,
        'Merseyside': 147564,
        'Norfolk': 57397,
        'North Yorkshire': 3123501,
        'Northamptonshire': 63375,
        'Northumberland': 88066,
        'Nottinghamshire': 77268,
        'Oxfordshire': 76155,
        'Rutland': 57398,
        'Shropshire': 57511,
        'Somerset': 3125930,
        'South Yorkshire': 88078,
        'Staffordshire': 57515,
        'Suffolk': 28595,
        'Surrey': 57582,
        'Tyne and Wear': 154376,
        'Warwickshire': 57516,
        'West Midlands': 57517,
        'West Sussex': 113757,
        'West Yorkshire': 88079,
        'Wiltshire': 2694420,
        'Worcestershire': 57581,

        # Welsh counties
        'Anglesey':         360939,
        'Brecknockshire':	359909,
        'Caernarvonshire':	298872,
        'Cardiganshire':	361613,
        'Carmarthenshire':	361616,
        'Denbighshire':	    298843,
        'Flintshire':	    298834,
        'Glamorgan':	    359902,
        'Merionethshire':	298875,
        'Monmouthshire':	359815,
        'Montgomeryshire':	298880,
        'Pembrokeshire':	361615,
        'Radnorshire':	    359950,
    }

    counties_gdf = ox.geocode_to_gdf([f'R{id}' for id in uk_counties.values()], by_osmid=True)
    return counties_gdf


def get_pc_bbox(db, north, south, east, west, min_year: Optional[int] = None, max_year: Optional[int] = None):
    df = db.get_pc_bbox(north, south, east, west, min_year=min_year, max_year=max_year)

    gdf = gpd.GeoDataFrame(df.drop(['longitude', 'latitude'], axis=1),
                           geometry=gpd.points_from_xy(df['longitude'], df['latitude']))
    gdf.crs = "EPSG:4326"

    # Encode categorical columns 
    gdf['new_build'] = (gdf['new_build_flag'] == 'Y').astype(int)
    gdf['freehold'] = (gdf['tenure_type'] == 'F').astype(int)
    gdf = gdf.join(pd.get_dummies(gdf['property_type']).astype(int).rename({
        'D': 'detached', 'S': 'semidetached', 'T': 'terraced', 'F': 'flat', 'O': 'other'
    }, axis=1))
    gdf = gdf.drop(['new_build_flag', 'tenure_type', 'property_type'], axis=1)

    return gdf

def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
