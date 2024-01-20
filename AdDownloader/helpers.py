"""This module provides different helper functions for the AdDownloader."""

import json
import pandas as pd
import os
from datetime import datetime
from prompt_toolkit.validation import Validator, ValidationError
import openpyxl


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
                cursor_position=len(document.text))  # Move cursor to end
        
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
                cursor_position=len(document.text))  # Move cursor to end

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
    Initialize the AdLibAPI object.

    :param file: A path to an excel file.
    :type file: str
    :returns: True if the string represents a valid path to an excel file, False otherwise.
    :rtype: bool
    """
    try:
        # check if the path exists and has an Excel file extension
        path = os.path.join("data", file)
        if not os.path.exists(path) or not path.lower().endswith(('.xlsx', '.xls', '.xlsm')):
            print(f"Excel file not found.")
            return False
        # try to read the excel file
        pd.read_excel(path)
        return True
    except Exception as e:  # catches any exception when trying to read
        return False


def load_json_from_folder(folder_path):
    """
    Load all the JSON files from the specified folder and merge then into a dataframe.

    :param file: A path to a folder containing JSON files with ad data.
    :type file: str
    :returns: A dataframe containing information retrieved from all JSON files of the folder.
    :rtype: pd.DataFrame
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


# function that flattens the age_country_gender_reach_breakdown column 
def flatten_age_country_gender(row, target_country):
    """
    Flatten an entry row containing the age_country_gender_reach_breakdown by putting it into wide format for a given target country.

    :param row: A row in JSON format containing age_country_gender_reach_breakdown data.
    :type row: list
    :param target_country: The target country for which the reach data will be processed.
    :type target_country: str
    :returns: A list with the processed age_gender_reach data.
    :rtype: list
    """
    flattened_data = []

    # Check if the row is empty and remove it
    if isinstance(row, float) and pd.isna(row):
        return flattened_data

    for entry in row:
        country = entry.get('country')
        if country in target_country: # take only the target country
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


def transform_data(project_name, country):
    """
    Transform all the data from a given project with a target country by flattening its age_country_gender_reach_breakdown column.
    This function will work if there exists a folder 'output/{project_name/json}' containing raw downloaded data in JSON format.
    The transformed data is saved inside 'output/{project_name}/ads_data', where original_data.xlsx is the original downloaded data and processed_data.xlsx contains flattened age_country_gender_reach_breakdown columns. 

    :param project_name: The name of the current project.
    :type file: str
    :param country: The target country for which the data will be transformed.
    :type country: str
    :returns: A dataframe with the processed age_gender_reach data.
    :rtype: pd.DataFrame
    """
    
    folder_path = f'output\\{project_name}\\json'
    df = load_json_from_folder(folder_path)

    # save original data
    data_path = f'output\\{project_name}\\ads_data'
    # check if the folder exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    df.to_excel(f'{data_path}\\original_data.xlsx', index=False)

    # flatten the age_country_gender_breakdown for each ad
    df['flattened_data'] = df['age_country_gender_reach_breakdown'].apply(flatten_age_country_gender, target_country=country)
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
    #TODO: remove index slicing here!!!
    final_data = df.iloc[:, :15].merge(wide_df, on="id")
    # fill the NAs in the reach columns
    selected_columns = [col for col in final_data.columns if col.startswith(('female', 'male', 'unknown'))]
    final_data[selected_columns] = final_data[selected_columns].fillna(0)

    final_data.to_excel(f'{data_path}\\processed_data.xlsx', index=False)
    # better use column names
    return final_data


def update_access_token(data, new_access_token=None):
    """
    Update the ad_snapshot_url with a new access token given ad data. 

    :param data: A dataframe containing a column ad_snapshot_url.
    :type data: pd.DataFrame
    :param new_access_token: The new access token, optional. If none is given, user will be prompted for inputting it.
    :type new_access_token: str
    :returns: A dataframe with the processed age_gender_reach data.
    :rtype: pd.DataFrame
    """
    if new_access_token is None:
        new_access_token = input("Please provide an update access token: ")
    data['ad_snapshot_url'] = data['ad_snapshot_url'].str.replace(r'access_token=.*$', f'access_token={new_access_token}', regex=True)
    return data


"""
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

"""