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

ExcelValidator Class
--------------------

.. autoclass:: ExcelValidator
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

      >>> row_example = [{"country": "NL", "age_gender_breakdowns": [{"age_range": "18-24", "male": 100, "female": 50, "unknown": 10}, ...]}]
      >>> target_country_example = "NL"
      >>> flattened_data = flatten_age_country_gender(row_example, target_country_example)
      >>> print(flattened_data)
      [{'country': 'NL', 'age_range': '18-24', 'male': 100, 'female': 50, 'unknown': 10}, ...]

flatten_demographic_distribution Function
-----------------------------------

.. autofunction:: flatten_demographic_distribution

   Example::

      >>> row_example = [{'percentage': '0.113043', 'age': '45-54', 'gender': 'male'}, {'percentage': '0.008696', 'age': '25-34', 'gender': 'female'}, ...]
      >>> flattened_data = flatten_demographic_distribution(row_example)
      >>> print(flattened_data)
      {'male_45-54': 0.113043, 'female_25-34': 0.008696, ...}

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

configure_logging Function
----------------------------

.. autofunction:: configure_logging

close_logger Function
----------------------------

.. autofunction:: close_logger

hide_access_token Function
----------------------------

.. autofunction:: hide_access_token

   Example::

      >>> data = pd.read_excel('path/to/your/data.xlsx')
      >>> data = hide_access_token(data)
      >>> data.to_excel('path/to/your/data.xlsx', index=False)

update_access_token Function
----------------------------

.. autofunction:: update_access_token

   Example::

      >>> data = pd.read_excel('path/to/your/data.xlsx')
      >>> new_access_token = input("Provide an updated access token: ")
      >>> data = update_access_token(data, new_access_token)

get_long_lived_token Function
----------------------------

.. autofunction:: get_long_lived_token

calculate_image_hash Function
-----------------------------

.. autofunction:: calculate_image_hash

   Example::

      >>> image_path = 'path-to-your-image'
      >>> calculate_image_hash(image_path)
      '108f46130f45639cf388892306235fd5'

deduplicate_images Function
---------------------------

.. autofunction:: deduplicate_images

   Example::

      >>> image_folder = 'output/<project_name>/ads_images'
      >>> unique_img_folder = 'output/<project_name>/unique_images'
      >>> deduplicate_images(image_folder, unique_img_folder)
      Found 57 duplicates and saved 143 unique images inside output/<project_name>/unique_images.