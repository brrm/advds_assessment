from .config import *

from . import access

import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import numpy as np
import geopandas as gpd
from geopandas.tools import sjoin, sjoin_nearest
from typing import Optional

from enum import Enum


class PropertyType(Enum):
    OTHER = 'other'
    FLAT = 'flat'
    TERRACED = 'terraced'
    SEMIDETACHED = 'semidetached'
    DETACHED = 'detached'

    def get_dummies(self):
        return {e.value: (1 if self.value == e.value else 0) for e in PropertyType}


def distance_decay(d, m=500, pow=4.0):
    """
    function from distance to closeness
    range is (0,1] with 1 for 0 distance
    :param m: distance at which closeness = 1/e
    :param pow: determines how aggressively closeness decays
    """

    return np.exp(-np.power((d / m), pow))


def calculate_zones(gdf, max_dist=500):
    """
    find what type of zone (land use) each point in gdf is
    :param gdf: GeoFrame with unique index
    :param max_dist: maximum distance (meters) to match zones
    """

    # gdf should have a geometry column and a unique index
    zone_types = ['commercial_retail', 'farmland', 'industrial', 'residential']

    west, east = gdf['geometry'].bounds['minx'].min(), gdf['geometry'].bounds['maxx'].max()
    south, north = gdf['geometry'].bounds['miny'].min(), gdf['geometry'].bounds['maxy'].max()

    zones = get_zones(north, south, east, west, buffer=0.01)

    # Lambert projection for distance calculations. Since distances are small, distortions should be negligible
    gdf_points = gdf[['geometry']].to_crs('EPSG:3347')
    zones = zones.to_crs('EPSG:3347')

    gdf_points = gdf_points.sjoin_nearest(zones[['geometry', 'zone']], how='left', max_distance=max_dist)
    gdf_points = gdf_points[~gdf_points.index.duplicated()]

    # Encode as categorical
    # Can't just use pd.get_dummies() because sometimes some values are not present
    gdf_zones = pd.DataFrame()
    for z in zone_types:
        gdf_zones[f'{z}_zone'] = (gdf_points['zone'] == z).astype(int)

    return gdf_zones


def calculate_poi_closeness(gdf, pois, features: list[str] = None, m=500, pow=4.0, max_dist=1500):
    """
    calculate closeness to POIs for each point in gdf
    :param gdf: GeoFrame with unique index
    :param pois: labelled POIs from get_pois_labels() to match on
    :param features: POI labels to calculate closeness for
    :param m: scale parameter for distance decay function
    :param pow: decay parameter for distance decay function
    :param max_dist: maximum distance (meters) to match POIs
    """

    # Lambert projection for distance calculations. Since distances are small, distortions should be negligible
    gdf_points = gdf[['geometry']].to_crs('EPSG:3347')
    pois = pois.to_crs('EPSG:3347')

    if features is None:
        features = ['school', 'healthcare', 'sustenance', 'entertainment', 'sports', 'public_transport',
                    'nature', 'water', 'park', 'food_shop', 'commerce', 'tourism']

    closeness_df = pd.DataFrame(index=gdf.index)
    for f in features:
        # Distance from each point in gdf to the nearest instance of feature f in pois
        dist = gdf_points.sjoin_nearest(pois.loc[pois[f], ['geometry', f]],
                                        how='left', max_distance=max_dist, distance_col='dist')
        dist = dist[~dist.index.duplicated()]  # equidistance results in duplicates so drop them

        closeness_df[f'{f}_closeness'] = distance_decay(dist['dist'].fillna(max_dist).values, m=m, pow=pow)

    return closeness_df


def calculate_poi_density(gdf, pois, features: list[str] = None, radius=500):
    """
    calculate number of POIs near each point in gdf
    :param gdf: GeoFrame with unique index
    :param pois: labelled POIs from get_pois_labels() to match on
    :param features: POI labels to calculate counts for
    :param radius: POI inclusion radius (meters) around points
    """

    # Lambert projection for distance calculations. Since distances are small, distortions should be negligible
    gdf_points = gdf[['geometry']].to_crs('EPSG:3347')
    gdf_points.geometry = gdf_points.geometry.buffer(radius)
    pois = pois.to_crs('EPSG:3347')

    if features is None:
        features = ['sustenance', 'entertainment', 'sports', 'public_transport',
                    'park', 'food_shop', 'commerce', 'tourism']

    density_df = pd.DataFrame()
    for f in features:
        # All instances of feature f in pois that are within radius of each point
        f_matches = gdf_points.sjoin_nearest(pois.loc[pois[f], ['geometry', f]],
                                             how='left', max_distance=radius, distance_col='dist')
        density_df[f'{f}_density'] = f_matches.groupby(f_matches.index)[f].count()

    return density_df


def plot_zones(ax, gdf, town: str = None, county: str = None, buffer=0.01):
    """
    Plots a map of zones and house prices
    :param ax: axes on which to plot
    :param gdf: GeoFrame with house prices
    :param town: optionally filter GeoFrame to only plot specific town
    :param county: optionally filter GeoFrame to only plot specific county
    :param buffer: margin (in degrees) around points to include
    """

    if not (county is None):
        gdf = gdf[gdf['county'] == county]

    if not (town is None):
        gdf = gdf[gdf['town_city'] == town]

    west, east = gdf['geometry'].bounds['minx'].min(), gdf['geometry'].bounds['maxx'].max()
    south, north = gdf['geometry'].bounds['miny'].min(), gdf['geometry'].bounds['maxy'].max()

    zones = get_zones(north, south, east, west, buffer=buffer)

    # Overpass graph
    graph = ox.graph_from_bbox(north + buffer, south - buffer, east + buffer, west - buffer)
    nodes, edges = ox.graph_to_gdfs(graph)

    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.set_xlim([west - buffer, east + buffer])
    ax.set_ylim([south - buffer, north + buffer])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"log-scale property prices & zones, {town}")

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['red', 'orange', 'blue', 'green'])
    zones.plot(ax=ax, column="zone", alpha=0.2, markersize=5, legend=True,
               cmap=cmap)

    # Geopandas' plot() has a bug for log normalisation, so we use scatter() instead
    res = ax.scatter(gdf.geometry.x, gdf.geometry.y, c=gdf['price'], cmap='magma', alpha=0.9, norm='log', s=4)
    plt.colorbar(res)
    return res


def plot_pois(ax, gdf, town: str = None, county: str = None, buffer=0.01):
    """
    Plots a map of POIs and house prices.
    Parks & nature are in green, water in blue, other POIs in yellow
    :param ax: axes on which to plot
    :param gdf: GeoFrame with house prices
    :param town: optionally filter GeoFrame to only plot specific town
    :param county: optionally filter GeoFrame to only plot specific county
    :param buffer: margin (in degrees) around points to include
    """

    if not (county is None):
        gdf = gdf[gdf['county'] == county]

    if not (town is None):
        gdf = gdf[gdf['town_city'] == town]

    west, east = gdf['geometry'].bounds['minx'].min(), gdf['geometry'].bounds['maxx'].max()
    south, north = gdf['geometry'].bounds['miny'].min(), gdf['geometry'].bounds['maxy'].max()

    pois = get_pois_labels(north, south, east, west, buffer=buffer)

    # Overpass graph
    graph = ox.graph_from_bbox(north + buffer, south - buffer, east + buffer, west - buffer)
    nodes, edges = ox.graph_to_gdfs(graph)

    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.set_xlim([west - buffer, east + buffer])
    ax.set_ylim([south - buffer, north + buffer])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"log-scale property prices & POIs, {town}")

    # Plot POIs
    pois[pois['park'] | pois['nature']].plot(ax=ax, color="green", alpha=0.2, markersize=5)
    pois[pois['water']].plot(ax=ax, color="blue", alpha=0.2, markersize=5)
    pois[~(pois['park'] | pois['water'] | pois['nature'])].plot(ax=ax, color="yellow", alpha=0.2, markersize=10)

    # Geopandas' plot() has a bug for log normalisation, so we use scatter() instead
    res = ax.scatter(gdf.geometry.x, gdf.geometry.y, c=gdf['price'], cmap='magma', alpha=0.9, norm='log', s=4)
    plt.colorbar(res)
    return res


def get_zones(north: float, south: float, east: float, west: float, buffer=0.01):
    """
    Get OpenStreetMap data of relevant zoning within bounding box
    We process data here (e.g. by merging zone types) hence why this is in `assess`
    """
    tags = {
        "landuse": ["commercial", "industrial", "residential", "retail", "farmland", "farmyard", "allotments"]
    }

    zones = ox.features_from_bbox(north + buffer, south - buffer, east + buffer, west - buffer, tags)
    # Group similar zones together
    zones['zone'] = zones['landuse'].replace({'commercial': 'commercial_retail', 'retail': 'commercial_retail',
                                              'farmyard': 'farmland', 'allotments': 'farmland'
                                              })

    return zones


def get_pois_labels(north: float, south: float, east: float, west: float, buffer=0.01, labels=None):
    """
    Get a selection of OpenStreetMap POIs within bounding box
    Relabeled so that returned GeoFrame only contains boolean cols
    indicating whether the associated POI is a specific type of POI (e.g. a school)
    """

    if labels is None:
        # Note we don't include buildings because they denote physical shape rather than function
        # e.g. "building=supermarket" means a building has the form of a typical supermarket building 
        # but does not indicate it has an active supermarket
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

            'food_shop': {
                'shop': ['bakery', 'butcher', 'cheese', 'deli', 'greengrocer', 'food', 'convenience', 'supermarket']},
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

    pois_raw = ox.features_from_bbox(north + buffer, south - buffer, east + buffer, west - buffer, tags)
    pois = pois_raw[['geometry', 'name']].copy()

    for label, label_tags in labels.items():
        # Conjunction of required tags
        flag = np.full(len(pois), False)
        for k, v in label_tags.items():
            if not (k in pois_raw.columns):
                continue

            if isinstance(v, bool):
                flag = flag | (~pois_raw[k].isna().values)
            else:
                flag = flag | pois_raw[k].isin(v).values

        pois[label] = flag

    return pois

def get_local_pois(gdf, buffer=0.01, labels=None):
    """
    calls get_pois_labels() using the min and max points in gdf as the bounding box
    """
    west, east = gdf['geometry'].bounds['minx'].min(), gdf['geometry'].bounds['maxx'].max()
    south, north = gdf['geometry'].bounds['miny'].min(), gdf['geometry'].bounds['maxy'].max()

    return get_pois_labels(north, south, east, west, buffer=buffer, labels=labels)

def plot_seasonal_prices(pp_seasonal_data):
    """
    Plots price vs year and a polar seasonal price chart for each year, side-by-side
    """
    pp_seasonal_data = pp_seasonal_data.set_index('year_of_transfer')
    # NB: yearly average is an approximation as we're implicitly assuming the same number of transactions each month
    pp_yearly_data = pp_seasonal_data.groupby(pp_seasonal_data.index)['avg_price'].mean()

    pp_seasonal_data['year_avg'] = pp_yearly_data
    # Price change vs year average in that month
    pp_seasonal_data['price_change'] = pp_seasonal_data['avg_price'] / pp_seasonal_data['year_avg'] - 1.0

    axs = (plt.subplot(121), plt.subplot(122, projection='polar'))
    axs[0].plot(pp_yearly_data)
    axs[0].set_xlabel('year')
    axs[0].set_ylabel('price')

    for year in pp_seasonal_data.index.unique():
        axs[1].plot(pp_seasonal_data.loc[year]['month_of_transfer'] - 1, pp_seasonal_data.loc[year]['price_change'])

    plt.thetagrids((0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330),
                   labels=('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'),
                   fmt=None)
    axs[1].set_theta_zero_location('N')
    axs[1].set_theta_direction(-1)
    axs[1].set_rticks([-0.05, 0, 0.05, 0.15], labels=['-5%', '0%', '5%', '15%'])
    axs[1].set_rlabel_position(-22.5)
    axs[1].set_title("% price change from year's average, each month")

    return axs


def plot_prices_counties(ax, gdf):
    '''
    Plots average price in gdf per UK Ceremonial County (OpenStreetMap data)
    '''

    counties_gdf = get_uk_counties_gdf()

    # Join counties with prices
    gdf = gdf.sjoin(counties_gdf[['geometry', 'name']], how='right')
    # Average price per county
    gdf = gdf.groupby('name').agg({'geometry': 'first', 'price': 'mean'}).set_geometry('geometry')
    gdf.crs = "EPSG:4326"  # Geometry gets reset on groupby so need to set again

    gdf['log_price'] = np.log10(gdf['price'])

    gdf.plot(ax=ax, column='log_price', edgecolor='black', cmap='plasma', legend=True)
    ax.set_title("log-scale mean transaction price by county")

    return ax


def plot_pc_category_counts(ax, gdf):
    '''
    Bar plot of counts of each category in `prices_coordinates_data`
    '''

    if not('property_type' in gdf.columns):
        gdf['property_type'] = gdf[['detached',	'semidetached', 'terraced', 'flat', 'other']].idxmax(axis=1)

    if not('tenure_type' in gdf.columns):
        gdf['tenure_type'] = np.where(gdf['freehold'], 'F', 'L')

    gdf_cats = gdf.groupby(['property_type', 'new_build'])['tenure_type'].value_counts()

    gdf_cats.unstack().plot(ax=ax, kind='barh', xlabel='count')

    return ax


def plot_zone_category_counts(ax, gdf):
    '''
    Bar plot of counts of each zone for each property type
    '''
    if not('property_type' in gdf.columns):
        gdf['property_type'] = gdf[['detached',	'semidetached', 'terraced', 'flat', 'other']].idxmax(axis=1)

    if not('zone' in gdf.columns):
        gdf['zone'] = gdf[['residential_zone', 'commercial_retail_zone',
                           'farmland_zone', 'industrial_zone']].idxmax(axis=1)

    gdf_cats = gdf.groupby('property_type')['zone'].value_counts()

    gdf_cats.unstack().plot(ax=ax, kind='barh', xlabel='count')

    return ax


def get_uk_counties_gdf():
    """
    Get OpenStreetMap UK Ceremonial County data
    """
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
        'Anglesey': 360939,
        'Brecknockshire': 359909,
        'Caernarvonshire': 298872,
        'Cardiganshire': 361613,
        'Carmarthenshire': 361616,
        'Denbighshire': 298843,
        'Flintshire': 298834,
        'Glamorgan': 359902,
        'Merionethshire': 298875,
        'Monmouthshire': 359815,
        'Montgomeryshire': 298880,
        'Pembrokeshire': 361615,
        'Radnorshire': 359950,
    }

    counties_gdf = ox.geocode_to_gdf([f'R{id}' for id in uk_counties_osmids.values()], by_osmid=True)
    return counties_gdf


def get_pc_bbox(db: access.DBConn, north, south, east, west,
                min_year: Optional[int] = None, max_year: Optional[int] = None, encode=True):
    """
    Get prices_coordinates data as in get_pc_bbox() in `address`
    Also converts to GeoFrame, and optionally one-hot encodes categorical columns
    """
    df = db.get_pc_bbox(north, south, east, west, min_year=min_year, max_year=max_year)

    gdf = gpd.GeoDataFrame(df.drop(['longitude', 'latitude'], axis=1),
                           geometry=gpd.points_from_xy(df['longitude'], df['latitude']))
    gdf.crs = "EPSG:4326"

    if encode:
        return encode_gdf(gdf)

    return gdf


def get_pc_location(db: access.DBConn, town: Optional[str] = None, county: Optional[str] = None,
                    min_year: Optional[int] = None, max_year: Optional[int] = None, encode=True):
    """
    Get prices_coordinates data as in get_pc_location() in `address`
    Also converts to GeoFrame, and optionally one-hot encodes categorical columns
    """
    df = db.get_pc_location(town=town, county=county, min_year=min_year, max_year=max_year)

    gdf = gpd.GeoDataFrame(df.drop(['longitude', 'latitude'], axis=1),
                           geometry=gpd.points_from_xy(df['longitude'], df['latitude']))
    gdf.crs = "EPSG:4326"

    if encode:
        return encode_gdf(gdf)

    return gdf


def encode_gdf(gdf):
    """
    One-hot encodes categorical columns in a GeoFrame corresponding to `prices_coordinates` data
    """
    # We can't just use pd.dummies() because sometimes some values aren't present
    gdf['new_build'] = (gdf['new_build_flag'] == 'Y').astype(int)
    gdf['freehold'] = (gdf['tenure_type'] == 'F').astype(int)
    gdf['detached'] = (gdf['property_type'] == 'D').astype(int)
    gdf['semidetached'] = (gdf['property_type'] == 'S').astype(int)
    gdf['terraced'] = (gdf['property_type'] == 'T').astype(int)
    gdf['flat'] = (gdf['property_type'] == 'F').astype(int)
    gdf['other'] = (gdf['property_type'] == 'O').astype(int)

    gdf['month_of_transfer'] = pd.to_datetime(gdf['date_of_transfer']).dt.month

    gdf = gdf.drop(['new_build_flag', 'tenure_type', 'property_type'], axis=1)

    return gdf