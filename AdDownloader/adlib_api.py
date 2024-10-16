"""This module provides the call to the Meta Ad Library API for ad data retrieval."""

import pandas as pd
from collections.abc import Mapping
import requests
import os

from datetime import datetime
from AdDownloader.helpers import *

class AdLibAPI:
    """A class representing the Meta Online Ad Library API connection point."""

    def __init__(self, access_token, version = "v20.0", project_name = datetime.now().strftime("%Y%m%d%H%M%S")):
        """
        Initialize the AdLibAPI object by providing a valid Meta developer token and a project name.

        :param access_token: The access token for authentication.
        :type access_token: str
        :param version: The version of the Meta Ad Library API. Default is "v18.0".
        :type version: str
        :param project_name: The name of the project. Default is the current date and time.
        :type project_name: str
        """

        self.version = version
        self.access_token = access_token
        self.base_url = "https://graph.facebook.com/{version}/ads_archive".format(version = self.version)
        self.fields = None
        self.request_parameters = {}
        self.project_name = project_name

        # create logger based on the project name
        self.logger = configure_logging(project_name)


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
        self.logger.info(f"Starting reading page {page_number}")
        response = requests.get(url, params = params)
        try:
            data = response.json()
            
        except Exception as e:
            print(f"Error ({type(e).__name__} - {str(e)}) occured on page {page_number}: {response}. Retrying...")
            self.logger.error(f"Error ({type(e).__name__} - {str(e)}) occured on page {page_number}: {response}. Retrying...")
            try:
                # retry calling the API one more time
                response = requests.get(url, params = params)
                data = response.json()
            except Exception as e:
                print(f"Error ({type(e).__name__} - {str(e)}) occured on page {page_number}: {response}. Finishing the download.")
                self.logger.error(f"Error ({type(e).__name__} - {str(e)}) occured on page {page_number}: {response}. Finishing the download.")
                return

        # check if there was an error - print the message
        if "error" in data:
            print(f"No data on page {page_number}.\nError: {data['error']['message']}.")
            self.logger.error(f"No data on page {page_number}. Error: {data['error']['message']}.")
            return
        # no error but also no data - print the response
        elif not "data" in data:
            print(f"No data on page {page_number}: {response}.")
            self.logger.error(f"No data on page {page_number}: {response}.")
            return
        # check if the output json file is empty and return
        if not bool(data["data"]):
            print("Page", page_number, "is empty.")
            self.logger.warning(f"Page {page_number} is empty.")
            return
        
        folder_path = f"output/{self.project_name}/json"
        # check if the folder exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # save the data to a JSON file
        if page_ids is None:
            with open(f"{folder_path}/{page_number}.json", "w") as json_file:
                json.dump(data, json_file, indent = 4)
        else:
            with open(f"{folder_path}/{page_ids}_{page_number}.json", "w") as json_file:
                json.dump(data, json_file, indent = 4)
        

        # check if there is a next page and retrieve further data
        if "paging" in data and "next" in data["paging"]:
            next_page_url = data["paging"]["next"]
            self.fetch_data(next_page_url, params, page_ids, page_number + 1)


    def add_parameters(self, fields = None, ad_reached_countries = 'NL', ad_delivery_date_min = "2023-01-01", ad_delivery_date_max = datetime.today().strftime('%Y-%m-%d'),
                       search_page_ids = None, search_terms = None, ad_type = "ALL", **kwargs):
        """
        Add parameters for the API request. Mandatory parameters are reached countries, start and end date, and either page_ids or search_terms.
        See available parameters here: https://developers.facebook.com/docs/marketing-api/reference/ads_archive/

        :param fields: The fields to include in the API response. Default is None, fields are retrieved from the created AdLibApi object.
        :type fields: str
        :param ad_reached_countries: The reached country for ad targeting. Default is 'NL'.
        :type ad_reached_countries: str
        :param ad_delivery_date_min: The minimum start date of ad delivery. Default is "2023-01-01".
        :type ad_delivery_date_min: str
        :param ad_delivery_date_max: The maximum start date of ad delivery. Default is the current date.
        :type ad_delivery_date_max: str
        :param search_page_ids: The name of the file containing page IDs. Default is None. Complementary with search_terms.
        :type search_page_ids: str
        :param search_terms: The search terms for ad filtering, in one string separated by a comma. Default is None. Complementary with search_page_ids.
        :type search_terms: str
        :param ad_type: The type of the ads to be retrieved. Default is "ALL", can also be "POLITICAL_AND_ISSUE_ADS".
        :type ad_type: str
        :param kwargs**: Add additional parameters for the search query, e.g. "estimated_audience_size_max = 10000"
        """

        if fields is None:
            fields = self.get_fields(ad_type)
            
        # check if the dates are valid    	
        if ad_delivery_date_min > datetime.today().strftime('%Y-%m-%d'):
            ad_delivery_date_min = datetime.today().strftime('%Y-%m-%d')
            print('Minimum delivery date is greater than the current date. Setting it as the current date.')
            self.logger.warning('Minimum delivery date is greater than the current date. Setting it as the current date.')
        
        if ad_delivery_date_max > datetime.today().strftime('%Y-%m-%d'):
            ad_delivery_date_max = datetime.today().strftime('%Y-%m-%d')
            print('Maximum delivery date is greater than the current date. Setting it as the current date.')
            self.logger.warning('Maximum delivery date is greater than the current date. Setting it as the current date.')
            
        if ad_delivery_date_min > ad_delivery_date_max:
            print('Minimum delivery date is greater than maximum delivery date. Swithching the dates around.')
            self.logger.warning('Minimum delivery date is greater than maximum delivery date. Swithching the dates around.')
            temp_min = ad_delivery_date_min
            ad_delivery_date_min = ad_delivery_date_max
            ad_delivery_date_max = temp_min
        
        params = {
            "fields": fields,
            "ad_reached_countries": ad_reached_countries,
            "ad_type": ad_type,
            "search_page_ids": None,
            "search_terms": None,
            "ad_delivery_date_min": ad_delivery_date_min,
            "ad_delivery_date_max": ad_delivery_date_max,
            "limit": "300",
            "access_token": self.access_token
        }

        # accept additional parameters through kwargs**
        params.update(kwargs)

        # search page ids - the file must contain at least one column called page_id
        if search_page_ids is not None:
            if is_valid_excel_file(search_page_ids):
                path = os.path.join("data", search_page_ids)
                try:
                    data = pd.read_excel(path)
                except:
                    try:
                        data = pd.read_csv(path)
                    except:
                        print('Unable to load page ids data.')
                        self.logger.error('Unable to load page ids data.')
                try:
                    search_page_ids_list = data['page_id'].astype(str).tolist()
                    params["search_page_ids"] = search_page_ids_list
                    self.request_parameters = params
                except:
                    print('Unable to read the page ids. Check if there exists a column `page_id` in your data.')
                    self.logger.error('Unable to read the page ids. Check if there exists a column `page_id` in your data.')
            else:
                print(f"Excel file not found.")

        elif search_terms is not None:
            params["search_terms"] = search_terms
            self.request_parameters = params

        else:
            print('You need to specify either pages ids or search terms.')
            self.logger.warning('You need to specify either pages ids or search terms.')

        self.logger.info(f'You added the following parameters: {self.request_parameters}')
            
    
    def start_download(self, params = None):
        """
        Start the download process from the Meta Ad Library API based on the provided parameters.

        :param params: The parameters for the API request. Default is None, parameters are retrieved from the created AdLibApi object.
        :type params: dict
        :returns: A dataframe containing the downloaded and processed ad data from the Meta Online Ad Library.
        :rtype: pandas.Dataframe
        """

        if params is None:
            params = self.request_parameters

        if params["search_terms"] is not None:
            self.fetch_data(url = self.base_url, params = params, page_number = 1)
        
        if params["search_page_ids"] is not None:
            search_page_ids_list = params["search_page_ids"]
            for i in range(0, len(search_page_ids_list), 10):                
                end_index = min(i + 9, len(search_page_ids_list) - 1)
                print(f"Fetching data starting for indexes [{i},{end_index+1}]")
                params["search_page_ids"] = str(search_page_ids_list[i:end_index+1])

                # call the function with the initial API endpoint and parameters
                self.fetch_data(self.base_url, params, page_ids = f"[{i},{end_index}]", page_number = 1)
        
        if not os.path.exists(f"output/{self.project_name}/json"):
            print("JSON files were not downloaded. Try a new request.")
            self.logger.info("JSON files were not downloaded. Try a new request.")
            return None
            
        nr_json_files = len([file for file in os.listdir(f"output/{self.project_name}/json") if file.endswith('.json')])
        print(f"Done downloading {nr_json_files} json files for the given parameters.")
        self.logger.info(f'Done downloading {nr_json_files} json files for the given parameters.')
        print("Data processing will start now.")
        self.logger.info('Data processing will start now.')

        # process into excel files:
        try:
            final_data = transform_data(self.project_name, country = params["ad_reached_countries"], ad_type = params["ad_type"])
            total_ads = len(final_data)
            print(f"Done processing and saving ads data for {total_ads} ads for project {self.project_name}.")
            self.logger.info(f'Done processing and saving ads data for {total_ads} ads for project {self.project_name}.')

            return(final_data)

        except Exception:
            print("No data was downloaded. Please try a new request.")
            self.logger.warning('No data was downloaded. Please try a new request.')
            
        # close the logger
        close_logger(self.logger)


    def get_parameters(self):
        """
        Get the parameters used for the API request (without the access token).

        :returns: A dictionary containing the parameters for the API request.
        :rtype: dict
        """
        params = self.request_parameters.copy()
        # remove the access_token from the copy
        params.pop("access_token", None)

        return(params)

    
    def clear_parameters(self):
        """
        Clear the current list of search parameters.]
        """
        self.request_parameters = {}
        self.logger.warning('Seach parameters removed.')
    

    def get_fields(self, ad_type):
        """
        Get the default fields for the API request, depends on the type of ads to be retrieved (All or Political). For available fields visit https://www.facebook.com/ads/library/api 

        :param ad_type: The type of the ads to be retrieved.
        :type ad_type: str
        :returns: A string containing the fields for the API request.
        :rtype: str
        """
        if ad_type == "ALL":
            return("id, ad_delivery_start_time, ad_delivery_stop_time, ad_creative_bodies, ad_creative_link_captions, ad_creative_link_descriptions, ad_creative_link_titles, ad_snapshot_url, beneficiary_payers, languages, page_id, page_name, target_ages, target_gender, target_locations, eu_total_reach, age_country_gender_reach_breakdown")
        else:
            return("id, ad_delivery_start_time, ad_delivery_stop_time, ad_creative_bodies, ad_creative_link_captions, ad_creative_link_descriptions, ad_creative_link_titles, ad_snapshot_url, bylines, currency, delivery_by_region, demographic_distribution, estimated_audience_size, impressions, languages, spend, page_id, page_name, target_ages, target_gender, target_locations")
