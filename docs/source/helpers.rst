Helpers Module
==============

.. module:: AdDownloader.helpers
   :synopsis: Provides different helper functions for the AdDownloader.

   This module provides different helper functions for the AdLibAPI object, such as validation and data processing tasks.

NumberValidator Class
---------------------

.. autoclass:: NumberValidator
   :members:

DateValidator Class
-------------------

.. autoclass:: DateValidator
   :members:

CountryValidator Class
----------------------

.. autoclass:: CountryValidator
   :members:

is_valid_excel_file Function
----------------------------

.. autofunction:: is_valid_excel_file

   Example::
   
      >>> is_valid_excel_file("example.xlsx")
      True

load_json_from_folder Function
------------------------------

.. autofunction:: load_json_from_folder

   Example::
   
      >>> folder_path = 'path/to/json/folder'
      >>> loaded_data = load_json_from_folder(folder_path)
      >>> print(loaded_data.head())

flatten_age_country_gender Function
-----------------------------------

.. autofunction:: flatten_age_country_gender

   Example::

      >>> row_example = [{"country": "NL", "age_gender_breakdowns": [{"age_range": "18-24", "male": 100, "female": 50, "unknown": 10}]}]
      >>> target_country_example = "NL"
      >>> flattened_data = flatten_age_country_gender(row_example, target_country_example)
      >>> print(flattened_data)
      [{'country': 'NL', 'age_range': '18-24', 'male': 100, 'female': 50, 'unknown': 10}]

transform_data Function
-----------------------

.. autofunction:: transform_data

   Example::

      >>> project_name_example = "example_project"
      >>> country_example = "NL"
      >>> transformed_data = transform_data(project_name_example, country_example, "ALL")
      >>> print(transformed_data.head())
            id ad_delivery_start_time ad_delivery_stop_time  ... unknown_45-54 unknown_55-64 unknown_65+
      0  11111             2023-12-21            2023-12-21  ...           0.0           0.0         0.0

      [1 rows x 33 columns]

update_access_token Function
----------------------------

.. autofunction:: update_access_token

   Example::

      >>> data = pd.read_excel('path/to/your/data.xlsx')
      >>> new_access_token = input("Provide an updated access token: ")
      >>> data = update_access_token(data, new_access_token)