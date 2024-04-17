"""This module provides different helper functions for the AdDownloader."""

import json
import pandas as pd
import os
import math
from datetime import datetime, timedelta
from prompt_toolkit.validation import Validator, ValidationError
import logging
import ast
import requests


class NumberValidator(Validator):
    """A class representing a number validator."""
    def validate(self, document):
        """
        Checks whether the input is a valid number.

        :param document: A document representing user's number input.
        :type document: document
        :returns: True if the text of the document represents a valid number, False otherwise.
        :rtype: bool
        """
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message='Please enter a number',
                cursor_position=len(document.text))  # move cursor to end
        
class DateValidator(Validator):
    """A class representing a date validator."""
    def validate(self, document):
        """
        Checks whether the input is a valid date in the format Y-m-d (e.g. "2023-12-31").

        :param document: A document representing user's date input.
        :type document: document
        :returns: True if the text of the document represents a valid date, False otherwise.
        :rtype: bool
        """
        try:
            datetime.strptime(document.text, '%Y-%m-%d')
        except ValueError:
            raise ValidationError(
                message='Please enter a valid date',
                cursor_position=len(document.text))

class CountryValidator(Validator):
    """A class representing a country code validator."""
    def validate(self, document):
        """
        Checks whether the input is a valid country code.

        :param document: A document representing user's country code input.
        :type document: document
        :returns: True if the text of the document represents a valid country code, False otherwise.
        :rtype: bool
        """
        
        country_codes = """ALL, BR, IN, GB, US, CA, AR, AU, AT, BE, CL, CN, CO, HR, DK, DO, EG, FI, FR, 
                DE, GR, HK, ID, IE, IL, IT, JP, JO, KW, LB, MY, MX, NL, NZ, NG, NO, PK, PA, PE, PH, 
                PL, RU, SA, RS, SG, ZA, KR, ES, SE, CH, TW, TH, TR, AE, VE, PT, LU, BG, CZ, SI, IS, 
                SK, LT, TT, BD, LK, KE, HU, MA, CY, JM, EC, RO, BO, GT, CR, QA, SV, HN, NI, PY, UY, 
                PR, BA, PS, TN, BH, VN, GH, MU, UA, MT, BS, MV, OM, MK, LV, EE, IQ, DZ, AL, NP, MO, 
                ME, SN, GE, BN, UG, GP, BB, AZ, TZ, LY, MQ, CM, BW, ET, KZ, NA, MG, NC, MD, FJ, BY, 
                JE, GU, YE, ZM, IM, HT, KH, AW, PF, AF, BM, GY, AM, MW, AG, RW, GG, GM, FO, LC, KY, 
                BJ, AD, GD, VI, BZ, VC, MN, MZ, ML, AO, GF, UZ, DJ, BF, MC, TG, GL, GA, GI, CD, KG, 
                PG, BT, KN, SZ, LS, LA, LI, MP, SR, SC, VG, TC, DM, MR, AX, SM, SL, NE, CG, AI, YT, 
                CV, GN, TM, BI, TJ, VU, SB, ER, WS, AS, FK, GQ, TO, KM, PW, FM, CF, SO, MH, VA, TD, 
                KI, ST, TV, NR, RE, LR, ZW, CI, MM, AN, AQ, BQ, BV, IO, CX, CC, CK, CW, TF, GW, HM, 
                XK, MS, NU, NF, PN, BL, SH, MF, PM, SX, GS, SS, SJ, TL, TK, UM, WF, EH"""
        country_codes = [code.strip() for code in country_codes.split(",")]
        ok = document.text in country_codes
        if not ok:
            raise ValidationError(
                message='Please enter a valid country code.',
                cursor_position=len(document.text))


def is_valid_excel_file(file):
    """
    Checks whether the input file name is a valid excel/csv file.

    :param file: A path to an excel file.
    :type file: str
    :returns: True if the string represents a valid path to an excel file, False otherwise.
    :rtype: bool
    """
    try:
        # check if the path exists and has an Excel file extension
        path = os.path.join("data", file)
        if not os.path.exists(path) or not path.lower().endswith(('.xlsx', '.xls', '.xlsm', '.csv')):
            print(f"Excel file not found.")
            return False
        # try to read the excel file
        #pd.read_excel(path)
        return True
    except Exception as e:  # catch any exception when trying to read
        return False


def load_json_from_folder(folder_path):
    """
    Load all the JSON files from the specified folder and merge then into a dataframe.

    :param file: A path to a folder containing JSON files with ad data.
    :type file: str
    :returns: A dataframe containing information retrieved from all JSON files of the folder.
    :rtype: pandas.DataFrame
    """
    # get a list of all files in the specified folder
    all_files = os.listdir(folder_path)
    
    # filter only files with a JSON extension
    json_files = [file for file in all_files if file.endswith('.json')]
    dfs = []
    # loop through each JSON file
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)

        # open the JSON file and read the content
        with open(file_path, 'r') as file:
            json_data = file.read()

        # parse and extract the data
        parsed_data = json.loads(json_data)
        data_list = parsed_data.get('data', [])

        df = pd.DataFrame(data_list)
        dfs.append(df)

    # concatenate all data frames a single one
    result_df = pd.concat(dfs, ignore_index=True)
    
    return result_df


def flatten_age_country_gender(row, target_country):
    """
    Flatten an entry row containing the age_country_gender_reach_breakdown by putting it into wide format for a given target country.

    :param row: A row in JSON format containing `age_country_gender_reach_breakdown` data.
    :type row: list
    :param target_country: The target country for which the reach data will be processed.
    :type target_country: str
    :returns: A list with the processed age_gender_reach data.
    :rtype: list
    """
    flattened_data = {}

    # check if the row is empty and remove it
    if isinstance(row, float) and pd.isna(row):
        return flattened_data
    
    # if row is a string (after loading the data from an Excel file), safely convert it back to a list
    if isinstance(row, str):
        try:
            row = ast.literal_eval(row)
        except (ValueError, SyntaxError):
            # in case the string cannot be converted back to a list
            return flattened_data

    for entry in row:
        country = entry.get('country')
        if country in target_country: # take only the target country
            age_gender_data = entry.get('age_gender_breakdowns', [])
            for age_gender_entry in age_gender_data:
                # exclude entries with 'Unknown' age range
                if age_gender_entry.get('age_range', '').lower() != 'unknown':
                    age_range = age_gender_entry['age_range']
                    male_count = age_gender_entry.get('male', 0)
                    female_count = age_gender_entry.get('female', 0)
                    unknown_count = age_gender_entry.get('unknown', 0)

                    # add all the entries to the flattened data                   
                    flattened_data[f"{country}_{age_range}_male"] = male_count
                    flattened_data[f"{country}_{age_range}_female"] = female_count
                    flattened_data[f"{country}_{age_range}_unknown"] = unknown_count

    return flattened_data


def flatten_demographic_distribution(row):
    """
    Flatten the demographic distribution data from a single row into a dictionary.

    This function takes a single row of demographic distribution data, which is typically a list of dictionaries containing percentage, age, and gender information. It flattens this nested structure into a dictionary with keys formatted as "{gender}_{age}" and corresponding percentage values.

    :param row: A row of demographic distribution data, typically a list of dictionaries.
    :type row: list
    :returns: A list where keys are formatted as "{gender}_{age}" and values are the corresponding percentage values.
    :rtype: list
    """
    flattened_data = {}
    if isinstance(row, float) and pd.isna(row):
        return flattened_data

    for entry in row:
        key = f"{entry['gender']}_{entry['age']}"
        flattened_data[key] = float(entry['percentage'])

    return flattened_data


def transform_data(project_name, country, ad_type):
    """
    Transform all the data from a given project with a target country by flattening its age_country_gender_reach_breakdown column.
    This function will work if there exists a folder 'output/{project_name/json}' containing raw downloaded data in JSON format.
    The transformed data is saved inside 'output/{project_name}/ads_data', where original_data.xlsx is the original downloaded data and processed_data.xlsx contains flattened age_country_gender_reach_breakdown columns. 

    :param project_name: The name of the current project.
    :type project_name: str
    :param country: The target country for which the data will be transformed.
    :type country: str
    :param ad_type: The type of the ads that were retrieved (can be "All" or "Political"). Depending on the `ad_type` different processing will be done.
    :type ad_type: str
    :returns: If ad_type = "All" then a dataframe with the processed age_country_gender_reach_breakdown data, if not then a dataframe with the processed demographic_distribution.
    :rtype: pandas.DataFrame
    """

    folder_path = f'output/{project_name}/json'
    df = load_json_from_folder(folder_path)

    # save original data
    data_path = f'output/{project_name}/ads_data'
    # check if the folder exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # save the original data as it came from the API
    df.to_excel(f'{data_path}/{project_name}_original_data.xlsx', index=False)

    # flatten the reach data (age-country-gender or demographic-distribution)
    if ad_type == "ALL":
        wide_df = pd.DataFrame(df['age_country_gender_reach_breakdown'].apply(flatten_age_country_gender, target_country=country).tolist())
    else:
        wide_df = pd.DataFrame(df['demographic_distribution'].apply(flatten_demographic_distribution).tolist())
 
        # create new columns with average impressions and spend
        df['impressions'] = df['impressions'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df['impressions_avg'] = df['impressions'].apply(lambda x: math.ceil((int(x['lower_bound']) + int(x.get('upper_bound', x['lower_bound']))) / 2))
        df['spend'] = df['spend'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df['spend_avg'] = df['spend'].apply(lambda x: math.ceil((int(x['lower_bound']) + int(x.get('upper_bound', x['lower_bound']))) / 2))
              
    # reorder the columns alphabetically and save the processed data to a different file
    wide_df = wide_df.reindex(sorted(wide_df.columns), axis=1)
    final_data = pd.concat([df, wide_df], axis=1)
    
    final_data_censored = hide_access_token(final_data)
    final_data_censored.to_excel(f'{data_path}/{project_name}_processed_data.xlsx', index=False)
    return final_data


def configure_logging(project_name):
    """
    Configures and returns a logger with a file handler set to write logs to a specified project's log file. 
    This function creates a log file named 'logs.log' within a directory named after the `project_name` under the 'output' directory.
    It checks if the logger already has handlers to prevent adding multiple handlers that do the same thing, ensuring 
    that each message is logged only once.

    :param project_name: The name of the project for which logging is being configured.
    :type project_name: str
    :returns: A configured logger object that logs messages to 'output/<project_name>/logs.log'.
    :rtype: logging.Logger
    """

    log_path = f"output/{project_name}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = logging.getLogger(project_name)
    logger.setLevel(logging.DEBUG)

    log_file = logging.FileHandler(os.path.join(log_path, "logs.log"), 'a')  # append mode
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    log_file.setFormatter(formatter)

    # Check if the logger already has handlers to prevent adding multiple handlers that do the same thing
    if not logger.handlers:
        logger.addHandler(log_file)

    return logger


def close_logger(logger):
    """
    Closes all handlers of the specified logger to ensure proper release of file resources.

    :param logger: The logger instance whose handlers are to be closed.
    :type logger: logging.Logger
    """
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def hide_access_token(data):
    """
    Remove the access token from `ad_snapshot_url` column. This can be readded by calling `update_access_token()`. 

    :param data: A dataframe containing a column `ad_snapshot_url`.
    :type data: pandas.DataFrame
    :returns: A dataframe with the access token removed from the `ad_snapshot_url` column.
    :rtype: pandas.DataFrame
    """
    data_copy = data.copy()
    data_copy['ad_snapshot_url'] = data_copy['ad_snapshot_url'].str.replace(r'access_token=.*$', 'access_token={access_token}', regex=True)
    return data_copy


def update_access_token(data, new_access_token=None):
    """
    Update the `ad_snapshot_url` with a new access token given ad data. 

    :param data: A dataframe containing a column `ad_snapshot_url`.
    :type data: pandas.DataFrame
    :param new_access_token: The new access token, optional. If none is given, user will be prompted for inputting it.
    :type new_access_token: str
    :returns: A dataframe with an updated access token in the `ad_snapshot_url` column.
    :rtype: pandas.DataFrame
    """
    if new_access_token is None:
        new_access_token = input("Please provide an updated access token: ")

    data_copy = data.copy()
    data_copy['ad_snapshot_url'] = data_copy['ad_snapshot_url'].str.replace(r'access_token=.*$', f'access_token={new_access_token}', regex=True)
    return data_copy


def get_long_lived_token(access_token = None, app_id = None, app_secret = None, version = "v19.0"):
    """
    Generate a Meta long-lived access token, that lasts around 60 days, given a valid short-lived access token.
    The long-lived access token and the expiration time will be saved in a `meta_long_lived_token.txt` file. The `app_id` and `app_secret` can be found inside your app at https://developers.facebook.com/apps/.

    :param access_token: A valid access token, optional. If none is given, user will be prompted for inputting it.
    :type access_token: str
    :param app_id: A valid access token, optional. If none is given, user will be prompted for inputting it.
    :type app_id: str
    :param app_secret: A valid access token, optional. If none is given, user will be prompted for inputting it.
    :type app_secret: str
    """

    url = f"https://graph.facebook.com/{version}/oauth/access_token"
    if access_token is None:
        access_token = input("Please provide a valid access token: ")

    if app_id is None:
        app_id = input("Please provide a valid app ID: ")

    if app_secret is None:
        app_secret = input("Please provide a valid app secret: ")

    params = {
        "grant_type": "fb_exchange_token",
        "client_id": app_id,
        "client_secret": app_secret,
        "fb_exchange_token": access_token
    }

    response = requests.get(url, params=params)
    response = response.json()
    if not "access_token" in response:
        print("Error encountered while trying to generate a long-lived token.")
        return
    
    expires_in = timedelta(seconds = response["expires_in"])
    response["expires_in"] = str(expires_in)
    
    with open("meta_long_lived_token.txt", "w") as file:
        json.dump(response, file, indent = 4)
    
    print(f"Long-lived token generated and saved successfully inside {file.name}.")