#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from splot.mapping import vba_choropleth, vba_legend, mapclassify_bin
from shapely.geometry import Point, Polygon, LineString
import libpysal.weights as sw 
from esda.moran import Moran_Local, Moran
from splot.esda import lisa_cluster, plot_local_autocorrelation, plot_moran
import warnings
import census
import us
import rasterio
warnings.simplefilter("ignore") 


# In[2]:


from rasterio.plot import show
import requests
import rasterstats as rstats


# In[2]:


c = census.Census('bbf0f9cd4d3e9a9bb4c295578e2e6cb2765ced0d', year=2020)


# In[23]:


#test
c.acs5.state(('NAME', 'B19013_001E'), us.states.MD.fips)


# In[29]:


#access census data
#variable names: total, white alone, black alone, asian alone, median household income
#consider ratio of income to poverty level? B17002
acs_list = c.acs5.state_county_tract(('NAME', 'B02001_001E','B02001_002E', 'B02001_002E', 'B02001_005E','B19013_001E'), us.states.NY.fips, '005', census.ALL)


# In[26]:


nyc_acs = pd.DataFrame(acs_list)


# In[37]:


codes={'manhattan':'061', 'staten_island':'085','bronx':'005', 'queens':'081', 'brooklyn':'047'}
list_of_lists=[]
for i in codes:
    acs_list = c.acs5.state_county_tract(('NAME', 'B02001_001E','B02001_002E', 'B02001_003E', 'B02001_005E','B19013_001E'), us.states.NY.fips, codes[i], census.ALL)
    list_of_lists.append(acs_list)
nyc_acs=pd.concat([pd.DataFrame(j) for j in list_of_lists])  


# In[38]:


nyc_acs.columns=['name', 'total','white_alone', 'black_alone', 'asian_alone', 'med_income', 'state', 'county', 'tract']
nyc_acs['GEOID']=nyc_acs['state']+nyc_acs['county']+nyc_acs['tract']
nyc_acs['percent_white'] = 100* (nyc_acs['white_alone'].astype('int')/nyc_acs['total'].astype('int'))
nyc_acs['percent_black'] = 100* (nyc_acs['black_alone'].astype('int')/nyc_acs['total'].astype('int'))
nyc_acs['percent_asian'] = 100* (nyc_acs['asian_alone'].astype('int')/nyc_acs['total'].astype('int'))


# In[39]:


nyc_acs.head()


# # NDVI, NDWI

# In[3]:


with rasterio.open('S:/376/Spring22/jkchap16/sentinel9_19_21/red4.tiff') as src:
    red = src.read(1, window=rasterio.windows.from_bounds(-8266069.89886961,  4938302.3518047 , -8204449.4381029 ,
        4999279.62331638, src.transform))
    profile=src.profile

with rasterio.open('S:/376/Spring22/jkchap16/sentinel9_19_21/nir8.tiff') as src:
    nir = src.read(1, window=rasterio.windows.from_bounds(-8266069.89886961,  4938302.3518047 , -8204449.4381029 ,
        4999279.62331638, src.transform))

with rasterio.open('S:/376/Spring22/jkchap16/sentinel9_19_21/swir11.tiff') as src:
    swir = src.read(1, window=rasterio.windows.from_bounds(-8266069.89886961,  4938302.3518047 , -8204449.4381029 ,
        4999279.62331638, src.transform))


# In[4]:


def calc_ndvi(nir, red):
    '''Calculate NDVI from integer arrays'''
    nir = nir.astype(np.float)
    red = red.astype(np.float)
    ndvi = (nir - red) / (nir + red)
    return ndvi
def calc_ndwi(nir, swir):
    nir = nir.astype(np.float)
    red = red.astype(np.float)
    ndwi = (nir - swir) / (nir + swir)
    return ndwi


# In[5]:


ndvi=calc_ndvi(nir, red)


# In[21]:


np.sum(nir==0)


# In[20]:


np.sum(red==0)


# In[18]:


np.sum(np.isnan(nir))


# In[17]:


np.sum(np.isnan(red))


# In[16]:


np.sum(np.isnan(ndvi))


# In[55]:


rboo=(red==0)


# In[56]:


nboo= (nir==0)


# In[57]:


ndviboo= np.isnan(ndvi)


# In[65]:


d= np.logical_and(rboo, nboo, ndviboo)


# In[66]:


np.sum(d)


# In[73]:


plt.figure(figsize=(10,10))
plt.imshow(d, alpha=1, cmap='viridis', vmin=0, vmax=1)
plt.colorbar()


# In[74]:


plt.figure(figsize=(10,10))
plt.imshow(ndvi, cmap='RdYlGn',vmin=0,vmax=1) 
plt.colorbar()
plt.title('NDVI')
plt.xlabel('Column #')
plt.ylabel('Row #')


# In[ ]:


raster =  rasterio.open('S:/376/Spring22/jkchap16/sentinel9_19_21/red4.tiff')
pixelSizeX, pixelSizeY  = raster.res
print(pixelSizeX, pixelSizeY)


# In[35]:


transform=rasterio.transform.from_origin(-8266069.89886961, 4999279.62331638, 93.92582035682425, 93.9421667922144 )
with rasterio.open('nyc_ndvi1.tif', 'w', driver='GTiff', 
                   height=ndvi.shape[0], width=ndvi.shape[1], 
                   count=1, dtype=ndvi.dtype, 
                   crs='+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext  +no_defs',
                   transform=transform ) as output:
    output.write(ndvi,1) 
    output.close()  


# In[36]:


with rasterio.open('S:/376/Spring22/jkchap16/nyc_ndvi.tif') as src:
    ndvi2 = src.read(1)
    affine=src.transform
    print(src.profile)


# In[37]:


plt.imshow(ndvi2, cmap='RdYlGn')


# In[85]:


#consider using newer parks properties dataset: https://data.cityofnewyork.us/Recreation/Parks-Properties/enfh-gkve/data
parks=gpd.read_file('S:/376/Spring22/jkchap16/nyc_parks.zip')
parks = parks.to_crs(epsg='3857')
parks=parks[(parks['landuse']=='Neighborhood Park') | (parks['landuse']=='Community Park') |(parks['landuse']=='Nature Area') |(parks['landuse']=='Garden') | (parks['landuse']=='Flagship Park')]
parks=parks.reset_index().rename(columns={'index': 'park_subsection'})
parks['park_subsection']=parks['park_subsection'].astype(str)
parks['park_id']=parks['parknum'] + '_' + parks['park_subsection']
parks.head()


# In[41]:


ndvi3=rasterio.open('S:/376/Spring22/jkchap16/nyc_ndvi1.tif')


# In[42]:


show(ndvi3)


# In[43]:


fig, ax = plt.subplots()
rasterio.plot.show(ndvi3, cmap='RdYlGn', ax=ax)
parks.plot(ax=ax, facecolor='none', edgecolor='black', alpha=.5)


# In[49]:


parks_ndvi = rstats.zonal_stats(parks.geometry, 'S:/376/Spring22/jkchap16/nyc_ndvi1.tif', affine = affine)


# In[50]:


avg = [r['mean'] for r in parks_ndvi]

parks['avg_ndvi'] = avg


# In[68]:


parks


# In[53]:


plt.figure(figsize=(10,10))
parks.plot('avg_ndvi', cmap='RdYlGn', legend=True)
plt.show()


# In[58]:


parks['avg_ndvi'].min()


# # Number of children's play areas in a park

# In[86]:


cpas=pd.read_csv('S:/376/Spring22/jkchap16/child_play_areas.csv')


# In[87]:


cpas.head()


# In[88]:


cpas['geometry']=gpd.GeoSeries.from_wkt(cpas['point'])
cpas_gdf=gpd.GeoDataFrame(cpas, geometry='geometry', crs='EPSG:4269')
cpas_gdf=cpas_gdf.to_crs(epsg='3857')


# In[89]:


parks2=gpd.sjoin(parks, cpas_gdf)
parks2


# In[90]:


dfcount = parks2.groupby('park_id')['OMPpropID'].count().rename('cpa_count').reset_index()
dfcount


# In[91]:


parks=parks.merge(dfcount, how='left', on='park_id')
parks['cpa_count']=parks['cpa_count'].fillna(0)
parks.head()


# # Number of water fountains

# In[92]:


water_fountain=gpd.read_file('S:/376/Spring22/jkchap16/water_fountains.zip')
water_fountain=water_fountain.to_crs(epsg='3857')


# In[93]:


water_fountain.head()


# In[94]:


parks3=gpd.sjoin(parks, water_fountain)


# In[95]:


dfcount = parks3.groupby('park_id')['objectid'].count().rename('fountain_count').reset_index()
dfcount


# In[96]:


parks=parks.merge(dfcount, how='left', on='park_id')
parks['fountain_count']=parks['fountain_count'].fillna(0)
parks.head()


# # Number of Food Faclilities (No spatial info)

# In[102]:


#food= pd.read_json('S:/376/Spring22/jkchap16/DPR_Eateries_001.json')
#food.head()


# In[112]:


#food_by_park=food.groupby('park_id')['name'].count().rename('food_count').reset_index()
#food_by_park=food_by_park[['park_id', 'food_count']]

#food_by_park


# In[114]:


#parks=parks.merge(food_by_park, how='left', left_on='parknum', right_on='park_id')
#parks['food_count']=parks['food_count'].fillna(0)
#parks.head()


# # Number of Athletic Facilities

# In[97]:


athletic= pd.read_csv('S:/376/Spring22/jkchap16/Athletic_Facilities.csv')


# In[98]:


athletic.head()


# In[99]:


athletic['geometry1']=gpd.GeoSeries.from_wkt(athletic['multipolygon'])
athletic_gdf=gpd.GeoDataFrame(athletic, geometry='geometry1', crs='EPSG:4269')
athletic_gdf=athletic_gdf.to_crs(epsg='3857')
athletic_gdf['geometry']=athletic_gdf['geometry1'].centroid
athletic_gdf.drop(columns='geometry1', inplace=True)


# In[100]:


athletic_gdf


# In[101]:


parks4=gpd.sjoin(parks, athletic_gdf)
parks4


# In[102]:


dfcount = parks4.groupby('park_id')['SYSTEM'].count().rename('athletic_count').reset_index()
dfcount


# In[103]:


parks=parks.merge(dfcount, how='left', on='park_id')
parks['athletic_count']=parks['athletic_count'].fillna(0)
parks.head()


# # Area containing Forever Wild Natural Areas

# In[156]:


wild=gpd.read_file('S:/376/Spring22/jkchap16/NYC Parks Forever Wild.zip')
wild=wild.to_crs(epsg='3857')


# In[157]:


wild.head()


# In[162]:


wildparks=gpd.sjoin(wild, parks, how='within')
wildparks.head()


# In[163]:


wildparks['area']=wildparks.geometry.area
dfcount = wildparks.groupby('park_id')['area'].sum().rename('wild_area').reset_index()
dfcount


# In[164]:


parks=parks.merge(dfcount, how='left', on='park_id')
parks['wild_area']=parks['wild_area'].fillna(0)
parks.head()


# # Area containing community gardens

# In[170]:


garden=gpd.read_file('S:/376/Spring22/jkchap16/GreenThumb Garden Info.zip')
garden=garden.to_crs(epsg='3857')
garden.head()


# In[172]:


gardenparks=gpd.sjoin(garden, parks, predicate='within')


# In[173]:


gardenparks['area']=gardenparks.geometry.area
dfcount = gardenparks.groupby('park_id')['area'].sum().rename('garden_area').reset_index()
dfcount


# In[174]:


parks=parks.merge(dfcount, how='left', on='park_id')
parks['garden_area']=parks['garden_area'].fillna(0)
parks.head()


# In[ ]:





# In[ ]:





# # Within Walk to a park buffer

# In[165]:


walk2park=gpd.read_file('S:/376/Spring22/jkchap16/Walk-to-a-Park Service-area.zip')
walk2park.to_crs(epsg='3147')
#epsg in meters to make buffer...


# In[166]:


walk2park['bufferid']=walk2park['type']


# In[167]:


walk2park.loc[walk2park['type'] == '1/2-Mile', 'bufferid'] = 805
walk2park.loc[walk2park['type'] == '1/4-Mile', 'bufferid'] = 402


# In[168]:


for i in range(len(walk2park)):
    if walk2park['bufferid'].loc[i] == 805:
        walk2park['geometry'].loc[i]=walk2park['geometry'].loc[i].buffer(805)
    else:
         walk2park['geometry'].loc[i]=walk2park['geometry'].loc[i].buffer(402)


# In[169]:


walk2park.to_crs(epsg='3857')
walk2park


# In[ ]:


#how to dissolve with geopandas...
#continents = world.dissolve(by='continent')


# In[ ]:





# # Length of trails

# In[147]:


trails=pd.read_csv('S:/376/Spring22/jkchap16/Parks_Trails.csv')


# In[148]:


trails.head()


# In[149]:


trails['geometry']=gpd.GeoSeries.from_wkt(trails['SHAPE'])
trails_gdf=gpd.GeoDataFrame(trails, geometry='geometry', crs='EPSG:4269')
trails_gdf=trails_gdf.to_crs(epsg='3857')


# In[154]:


trails_gdf


# In[153]:


fig, ax=plt.subplots()
trails_gdf.plot(ax=ax, color='black')


# In[150]:


parks5=gpd.sjoin(trails_gdf, parks)
parks5


# In[ ]:


parks5['length']=parks5
dfcount = parks5.groupby('park_id')['SYSTEM'].count().rename('athletic_count').reset_index()
dfcount


# In[ ]:


parks=parks.merge(dfcount, how='left', on='park_id')
parks['athletic_count']=parks['athletic_count'].fillna(0)
parks.head()


# # Closest (subway station, bus stop, bike rack etc) to park center
