"""This module provides the call to the Meta Ad Library API for ad data retrieval."""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import os
from datetime import datetime
from AdDownloader.helpers import *

class AdLibAPI:

    def __init__(self, access_token, version = "v18.0"):
        self.version = version
        self.access_token = access_token
        #self.base_url = "https://graph.facebook.com/{version}/ads_archive?access_token={access_token}".format(version = self.version, access_token = self.access_token)
        self.base_url = "https://graph.facebook.com/{version}/ads_archive".format(version = self.version)
        self.fields = self.get_fields()
        self.request_parameters = {}
        self.project_name = None

    # function to fetch and process data based on url and params
    def fetch_data(self, url, params, page_ids = None, page_number = 1):
        print("##### Starting reading page", page_number, "#####")
        response = requests.get(url, params=params)
        data = response.json()

        # check if the output json file is empty and return
        if not "data" in data:
            print("No data on page", page_number)
            return
        if not bool(data["data"]):
            print("Page", page_number, "is empty.")
            return
        
        folder_path = f"output\\{self.project_name}\\json"
        # check if the folder exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # save the data to a JSON file
        if page_ids is None:
            with open(f"{folder_path}\\{page_number}.json", "w") as json_file:
                json.dump(data, json_file, indent=4)
        else:
            with open(f"{folder_path}\\{page_ids}_{page_number}.json", "w") as json_file:
                json.dump(data, json_file, indent=4)
        

        # check if there is a next page and retrieve further data
        if "paging" in data and "next" in data["paging"]:
            next_page_url = data["paging"]["next"]
            self.fetch_data(next_page_url, params, page_ids, page_number+1)


    
    def add_parameters(self, fields = None, countries = 'NL', start_date = "2023-01-01", end_date = datetime.today().strftime('%Y-%m-%d'),
                       page_ids = None, search_terms = None, project_name = datetime.now().strftime("%Y%m%d%H%M%S")):
        # see available parameters here: https://developers.facebook.com/docs/marketing-api/reference/ads_archive/
        if fields is None:
            fields = self.get_fields()

        self.project_name = project_name

        # some checks:
        # datetime format
        if not is_valid_date(start_date):
            print(f"The start date {start_date} is not valid. The default date 2023-01-01 will be used.")
            start_date = "2023-01-01"

        if not is_valid_date(end_date):
            print(f"The end date {end_date} is not valid. The default today's date will be used.")
            end_date = datetime.today().strftime('%Y-%m-%d')
        
        # countries
        countries = is_valid_country(countries)
        
        # page ids - the file must contain at least one column called page_id
        if page_ids is not None:
            if is_valid_excel_file(page_ids):
                path = os.path.join("data", page_ids)
                data = pd.read_excel(path)
                search_page_ids_list = data['page_id'].tolist()
                params = {
                    "fields": fields,
                    "ad_reached_countries": countries,
                    "search_page_ids": search_page_ids_list,
                    "search_terms": None,
                    "ad_delivery_date_min": start_date,
                    "ad_delivery_date_max": end_date,
                    "limit": "300",
                    "access_token": self.access_token
                }
                self.request_parameters = params

        elif search_terms is not None:
            params = {
                "fields": fields,
                "ad_reached_countries": countries,
                "search_page_ids": None,
                "search_terms": search_terms,
                "ad_delivery_date_min": start_date,
                "ad_delivery_date_max": end_date,
                "limit": "300",
                "access_token": self.access_token
            }
            self.request_parameters = params
            
    
    def start_download(self, params=None):
        if params is None:
            params = self.request_parameters

        if params["search_terms"] is not None:
            self.fetch_data(url=self.base_url, params=params, page_number=1)
        
        if params["search_page_ids"] is not None:
            search_page_ids_list = params["search_page_ids"]
            for i in range(0, len(search_page_ids_list), 10):
                end_index = min(i + 9, len(search_page_ids_list) - 1)
                print(f"Fetching data starting for indexes [{i},{end_index}]")
                params["search_page_ids"] = str(search_page_ids_list[i:end_index])

                # call the function with the initial API endpoint and parameters
                self.fetch_data(self.base_url, params, page_ids=f"[{i},{end_index}]", page_number=1)
        
        print(f"Done downloading json files for the given parameters.")


    def get_parameters(self):
        return(self.request_parameters)
    
    def get_fields(self):
        return("id, ad_creation_time, ad_delivery_stop_time, ad_creative_bodies, ad_creative_link_captions, ad_creative_link_descriptions, ad_creative_link_titles, ad_snapshot_url, page_id, page_name, target_ages, target_gender, target_locations, eu_total_reach, age_country_gender_reach_breakdown")


"""
####### put everything below in a loop to go through all categories:


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


"""