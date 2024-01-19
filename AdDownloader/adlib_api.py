"""This module provides the call to the Meta Ad Library API for ad data retrieval."""

import pandas as pd
import requests
import os
from datetime import datetime
from AdDownloader.helpers import *

class AdLibAPI:
    """A class representing the Meta Online Ad Library API connection point."""

    def __init__(self, access_token, version = "v18.0"):
        """
        Initialize the AdLibAPI object.

        :param access_token: The access token for authentication.
        :type access_token: str
        :param version: The version of the Meta Ad Library API. Default is "v18.0".
        :type version: str
        """
        
        self.version = version
        self.access_token = access_token
        self.base_url = "https://graph.facebook.com/{version}/ads_archive".format(version = self.version)
        self.fields = self.get_fields()
        self.request_parameters = {}
        self.project_name = None

    # function to fetch and process data based on url and params
    def fetch_data(self, url, params, page_ids = None, page_number = 1):
        """
        Fetch and process data based on the provided URL and parameters.

        :param url: The URL for making the API request.
        :type url: str
        :param params: The parameters to include in the API request.
        :type params: dict
        :param page_ids: Page IDs for naming output files. Default is None.
        :type page_ids: str
        :param page_number: The page number for tracking the progress. Default is 1.
        :type page_number: int
        """
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
        """
        Add parameters for the API request.
        See available parameters here: https://developers.facebook.com/docs/marketing-api/reference/ads_archive/

        :param fields: The fields to include in the API response. Default is None, fields are retrieved from the created AdLibApi object.
        :type fields: str
        :param countries: The country for ad targeting. Default is 'NL'.
        :type countries: str
        :param start_date: The start date for ad delivery. Default is "2023-01-01".
        :type start_date: str
        :param end_date: The end date for ad delivery. Default is the current date.
        :type end_date: str
        :param page_ids: The file containing page IDs. Default is None. Complementary with search_terms.
        :type page_ids: str
        :param search_terms: The search terms for ad filtering. Default is None. Complementary with page_ids.
        :type search_terms: str
        :param project_name: The name of the project. Default is the current date and time.
        :type project_name: str
        """
        if fields is None:
            fields = self.get_fields()

        self.project_name = project_name

        params = {
            "fields": fields,
            "ad_reached_countries": countries,
            "search_page_ids": None,
            "search_terms": None,
            "ad_delivery_date_min": start_date,
            "ad_delivery_date_max": end_date,
            "limit": "300",
            "access_token": self.access_token
        }

        #TODO: accept additional parameters through kwargs**
        # kwargs.update(fields = fields)
        # headers = kwargs
        # headers += ["&{key}={value}".format(key = str(key), value = str(value)) for key, value in kwargs.items()]
        # print(f"you added the following params: {headers}")

        # page ids - the file must contain at least one column called page_id
        if page_ids is not None:
            if is_valid_excel_file(page_ids):
                path = os.path.join("data", page_ids)
                data = pd.read_excel(path)
                search_page_ids_list = data['page_id'].tolist()
                params["search_page_ids"] = search_page_ids_list
                self.request_parameters = params

        elif search_terms is not None:
            params["search_terms"] = search_terms
            self.request_parameters = params

        else:
            print('You need to specify either pages ids or search terms.')
            
    
    def start_download(self, params=None):
        """
        Start the download process from the Meta Ad Library API based on the provided parameters.

        :param params: The parameters for the API request. Default is None, parameters are retrieved from the created AdLibApi object.
        :type params: dict
        :returns: A dataframe containing the downloaded and processed ad data from the Meta Online Ad Library.
        :rtype: pd.Dataframe
        """
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
        
        print("Done downloading json files for the given parameters.")
        print("Data processing will start now.")

        # process into excel files:
        try:
            final_data = transform_data(self.project_name, country=params["ad_reached_countries"]) 
            total_ads = len(final_data)
            print(f"Done processing and saving ads data for {total_ads} ads for project {self.project_name}.")
            return(final_data)

        except Exception:
            print("No data was downloaded. Please try a new request.")


    def get_parameters(self):
        """
        Get the parameters used for the API request.

        :returns: A dictionary containing the parameters for the API request.
        :rtype: dict
        """

        return(self.request_parameters)
    
    def get_fields(self):
        """
        Get the default fields for the API request.

        :returns: A string containing the fields for the API request.
        :rtype: str
        """
        #TODO: add different fields based on political ads (impressions, total_spend, etc.)
        # for available fields visit: https://developers.facebook.com/docs/marketing-api/reference/archived-ad/
        return("id, ad_delivery_start_time, ad_delivery_stop_time, ad_creative_bodies, ad_creative_link_captions, ad_creative_link_descriptions, ad_creative_link_titles, ad_snapshot_url, page_id, page_name, target_ages, target_gender, target_locations, eu_total_reach, age_country_gender_reach_breakdown")


"""

# alcohol nl: [882849606601570,306533725582037,2629981837151638,811933024017255]
# 882849606601570 - //*[@id="content"]/div/div/div/div/div/div/div[2]/div[2]/img
# 306533725582037 - //*[@id="content"]/div/div/div/div/div/div/div[2]/div[2]/img
# 2629981837151638 - //*[@id="content"]/div/div/div/div/div/div/div[2]/a/div[1]/img
# 811933024017255 


# 315337970888155, 942331563502762 for drinks be, //*[@id="content"]/div/div/div/div/div/div/div[2]/div[1]/video
# 186811847833644, 886890406228902 for drinks nl
# 737451568226648 - the weird fanta page


# update access token before downloading images
new_token = input("New access token:")
ads_data['ad_snapshot_url'] = ads_data['ad_snapshot_url'].str.replace(r'access_token=.*$', f'access_token={new_token}', regex=True)

"""