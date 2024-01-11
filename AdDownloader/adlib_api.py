"""This module provides the call to the Meta Ad Library API for ad data retrieval."""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import os
from helpers import *
from media_download import *

########## GET DATA USING THE REQUESTS LIBRARY:

url = "https://graph.facebook.com/v18.0/ads_archive"

# change the params below to retrieve different data/periods - only 10 page_ids at once

params = {
    "fields": "id, ad_creation_time, ad_delivery_stop_time, ad_creative_bodies, ad_creative_link_captions, ad_creative_link_descriptions, ad_creative_link_titles, ad_snapshot_url, page_id, page_name, target_ages, target_gender, target_locations, eu_total_reach, age_country_gender_reach_breakdown",
    "ad_reached_countries": None,
    "search_page_ids": None,
    "ad_delivery_date_min": "2023-01-01",
    "limit": "300",
    "access_token": None
}

params["access_token"] = input("Access token:")

# get all the names of all categories (by cat and country)
all_brands = os.listdir('data\\brands')
all_brands = [brand for brand in all_brands if brand != 'brands_all.xlsx']
categories = [os.path.splitext(name)[0].lower() for name in all_brands]

####### put everything below in a loop to go through all categories:
for j in range(6, 7):
    data = pd.read_excel(f"data/brands/{all_brands[j]}")
    data.head(10)
    search_page_ids_list = data['page_id'].tolist()

    for i in range(0, len(search_page_ids_list), 10):
        end_index = min(i + 9, len(search_page_ids_list) - 1)
        print(f"Fetching data starting for indexes [{i},{end_index}]")
        params["search_page_ids"] = str(search_page_ids_list[i:end_index])
        if categories[j].endswith("be"):
            params["ad_reached_countries"] = 'BE'
        else:
            params["ad_reached_countries"] = 'NL'
        # call the function with the initial API endpoint and parameters
        fetch_data(url, params, f"[{i},{end_index}]", page_number=1, cat=categories[j])

    print(f"Done downloading json files for {categories[j]}.")

    # load the data from the saved json files
    folder_path = f'data\\json\\{categories[j]}'
    final_data = transform_data(folder_path) 

    # Get all columns that start with 'female', 'male', or 'unknown'
    selected_columns = [col for col in final_data.columns if col.startswith(('female', 'male', 'unknown'))]

    final_data[selected_columns] = final_data[selected_columns].fillna(0)

    final_data.head(5)
    len(final_data)

    #final_data.to_csv(f'data\\ads\\{categories[0]}.csv', index=False)
    final_data.to_excel(f'data\\ads\\{categories[j]}.xlsx', index=False)
    print(f"Done fetching and saving ads data for {categories[j]}. Starting downloading media content.")

    # start media downloading here
    start_media_download(all_brands[j], is_file=True) # some bug here


# alcohol nl: [882849606601570,306533725582037,2629981837151638,811933024017255]
# 882849606601570 - //*[@id="content"]/div/div/div/div/div/div/div[2]/div[2]/img
# 306533725582037 - //*[@id="content"]/div/div/div/div/div/div/div[2]/div[2]/img
# 2629981837151638 - //*[@id="content"]/div/div/div/div/div/div/div[2]/a/div[1]/img
# 811933024017255 


# 315337970888155, 942331563502762 for drinks be, //*[@id="content"]/div/div/div/div/div/div/div[2]/div[1]/video
# 186811847833644, 886890406228902 for drinks nl
# 737451568226648 - the weird fanta page

category = all_brands[7]
file_path_ads = f'data\\ads\\{category}'
file_path_cat = f'data\\brands\\{category}'
ads_data = pd.read_excel(file_path_ads)
len(ads_data)
ads_data.head(5)

# count total nr of ads in all excel files
n = 0
for cat in all_brands:
    print(cat)
    file_path_ads = f'data\\ads\\{cat}'
    ads_data = pd.read_excel(file_path_ads)
    n += len(ads_data)
    print(len(ads_data))

print(f"Total number of ads collected is {n}.")

print(ads_data['ad_snapshot_url'][3])
# update access token before downloading images
new_token = input("New access token:")
ads_data['ad_snapshot_url'] = ads_data['ad_snapshot_url'].str.replace(r'access_token=.*$', f'access_token={new_token}', regex=True)

download_media_by_brand(category, is_file=False, data=ads_data)


# old fetching
data = pd.read_excel(f"data/brands/{all_brands[7]}")
data.head(10)
# Access the 'page_id' column as a list
search_page_ids_list = data['page_id'].tolist()

for i in range(0, len(search_page_ids_list), 10):
    end_index = min(i + 9, len(search_page_ids_list)-1)
    print(f"Fetching data starting for indexes [{i},{end_index}]")
    params["search_page_ids"] = str(search_page_ids_list[i:end_index])
    # call the function with the initial API endpoint and parameters
    fetch_data(url, params, f"[{i},{end_index}]", page_number=1, cat=categories[7])


print(f"Done downloading json files for {categories[7]}.")


# load the data from the saved json file - not very efficient ???
folder_path = f'data\\json\\{categories[7]}'
final_data = transform_data(folder_path) 
final_data.columns
final_data.head(5)
len(final_data)

# check for duplicates
len(final_data[final_data.duplicated(subset=['id'], keep=False)])
final_data[final_data.duplicated(subset=['id', 'ad_creation_time', 'ad_snapshot_url'], keep=False)]
final_data = final_data.drop_duplicates(subset=['id'])
len(final_data)
# 300 id duplicates, but with different urls

# fill NaNs with 0s
final_data.iloc[:, 14:] = final_data.iloc[:, 14:].fillna(0)

#final_data.to_csv(f'data\\ads\\{categories[0]}.csv', index=False)
final_data.to_excel(f'data\\ads\\{categories[7]}.xlsx', index=False)