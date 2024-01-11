"""This module provides different helper functions for the AdDownloader."""

import json
import numpy as np
import requests
import pandas as pd
import os

# function to fetch and process data based on url and params
def fetch_data(url, params, page_ids, page_number, cat):
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
    
    folder_path = f"data\\json\\{cat}"
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If it doesn't exist, create the folder
        os.makedirs(folder_path)

    # save the data to a JSON file
    with open(f"{folder_path}\\{page_ids}_{page_number}.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

    # check if there is a next page and retrieve further data
    if "paging" in data and "next" in data["paging"]:
        next_page_url = data["paging"]["next"]
        fetch_data(next_page_url, params, page_ids, page_number+1, cat)


########## JSON DATA PROCESSING
def load_json(file_path):
    # open the JSON file and read the content as text
    with open(file_path, 'r') as json_file:
        json_data = json_file.read()
    
    # parse and extract the data
    parsed_data = json.loads(json_data)
    data_list = parsed_data.get('data', [])
    len(data_list)
    df = pd.DataFrame(data_list)
    return df


def load_json_from_folder(folder_path):
    # Get a list of all files in the specified folder
    all_files = os.listdir(folder_path)
    
    # Filter only files with a JSON extension
    json_files = [file for file in all_files if file.endswith('.json')]
    dfs = []
    # Loop through each JSON file
    for json_file in json_files:
        # Construct the full file path
        file_path = os.path.join(folder_path, json_file)

        # Open the JSON file and read the content as text
        with open(file_path, 'r') as file:
            json_data = file.read()

        # Parse and extract the data
        parsed_data = json.loads(json_data)
        data_list = parsed_data.get('data', [])

        # Create a DataFrame from the current JSON data
        df = pd.DataFrame(data_list)

        # Append the DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    result_df = pd.concat(dfs, ignore_index=True)
    return result_df


# function that flattens the age_country_gender_reach_breakdown column 
def flatten_age_country_gender(row):
    flattened_data = []

    # Check if the row is empty and remove it
    if isinstance(row, float) and pd.isna(row):
        return flattened_data
    
    for entry in row:
        country = entry.get('country') # get the country to keep only BE or NL
        if country in ['BE', 'NL']: # maybe also adjust to take only the target country
            age_gender_data = entry.get('age_gender_breakdowns', [])
            for age_gender_entry in age_gender_data:
                # exclude entries with 'Unknown' age range
                if age_gender_entry.get('age_range', '').lower() != 'unknown':
                    # extract each field and flatten it together
                    flattened_entry = {
                        'country': country,
                        'age_range': age_gender_entry.get('age_range', ''),
                        'male': age_gender_entry.get('male', 0),
                        'female': age_gender_entry.get('female', 0),
                        'unknown': age_gender_entry.get('unknown', 0)
                    }
                    flattened_data.append(flattened_entry)
    return flattened_data

# indiv - procss only one file or entire folder
def transform_data(folder_path, indiv = False):
    if indiv:
        df = load_json(folder_path)
    else:
        df = load_json_from_folder(folder_path)

    # flatten the age_country_gender_breakdown for each ad
    df['flattened_data'] = df['age_country_gender_reach_breakdown'].apply(flatten_age_country_gender)
    # create a new DataFrame from the flattened data
    flattened_df = pd.DataFrame(df['flattened_data'].sum()) 

    # create a list of ids for the flattened data
    id_list = []
    for index, row in df.iterrows():
        id_list.extend([row['id']] * len(row['flattened_data']))
    flattened_df['id'] = id_list

    # convert to wide format
    wide_df = flattened_df.pivot_table(index=['id'], columns='age_range', values=['male', 'female', 'unknown'], aggfunc='first')
    # change the column names and reset the index
    wide_df.columns = ['_'.join(col) for col in wide_df.columns.values]
    wide_df.reset_index(inplace=True)

    # keep only the relevant columns and save data to csv
    final_data = df.iloc[:, :14].merge(wide_df, on="id")
    # better use column names
    return final_data

