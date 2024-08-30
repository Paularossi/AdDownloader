# AdDownloader
AdDownloader is a Python command-line tool designed for downloading ads and their associated media content from the Meta Ad Library. This tool provides an efficient and user-friendly way to access ad data for analysis and research purposes, offering additional analytical functionalities for the ad creatives.

*Key Features:*

* 🚀 Choose your fighter: easy-to-use Command-Line Interface or intuitive Python package.
* 🔐 Access Meta Ad Library data effortlessly by only providing your access token.
* 🔎 Answer questions based on your search target and download ads and their media content.
* 💡 Use the downloaded ad data for research and analysis of ad campaigns, embedded in a Dashboard.

Below, you can find information about the installing process and basic usage of AdDownloader. For a detailed description of the full features of this package, visit the official [AdDownloader documentation](https://addownloader.readthedocs.io/en/latest/index.html).

## Table of Contents
- [Introduction](#introduction)
    - [Meta Ad Library API](#meta-ad-libray-api)
    - [AdDownloader Functionalities](#addownloader-functionalities)
- [Prerequisities](#prerequisities)
- [Installation](#installation)
    - [From source or wheel-file](#from-source-or-wheel-file)
    - [From pip](#from-pip)
- [Usage](#usage)
    - [As a Python package](#as-a-python-package)
    - [As a CLI](#as-a-cli)
- [Contributing](#contributing)
- [License](#license)

## Introduction

### Meta Ad Library API

The <a href="https://www.facebook.com/ads/library/api/" target="_blank">Meta Ad Library API</a> is a feature of Meta's (formerly Facebook) extensive advertising transparency efforts. It allows programmatic access to data about ads run on Meta platforms (Facebook, Instagram, etc.), particularly focusing on social issues, elections, and politics. The Ad Library API helps you perform customised searches of the Ad Library for:
* Ads about social issues, election or politics that were delivered anywhere in the world during the past seven years
* Ads of any type that were delivered to the European Union during the past year.

Researchers are currently able to obtain the following data:

**All available ads will have:**
- Library ID
- Content of the ad creative (subject to our Terms of Service)
- Page name and Page ID associated with the ad
- Ad delivery dates
- Where the ad appeared (Facebook, Instagram etc.)

**Ads about social issues, elections or politics will also have:**
- Total amount spent (range)
- Total impressions received (range)
- Demographic information on ad reach, such as age, gender and location (%)

**Ads that were delivered to the European Union will also have:**
- Total impressions that an ad received in the EU (estimated)
- Targeting and reach demographic information specific to the EU (estimated)
- Beneficiary and payer information

For more information on getting access to the API, available fields, search parameters and API call examples, refer to the  <a href="https://www.facebook.com/ads/library/api/" target="_blank"> Meta Ad Library API Documentation</a>.

### AdDownloader functionalities
The AdDownloader has two main functionalities: (1) download general ad data based on user-inputed parameters; and (2) download media content for ads

Currently, Meta doesn't provide ad media content directly from an API call, but permits downloading individual ad creatives such as images or videos, for academic research purposes. This can be achieved by accessing the `ad_snapshot_url` from downloaded ad data and then manually retrieving the media content from the page. Addownloader automates this process for you! Briefly, the workflow can be described as follows:

1. 🎯 **Choose Your Desired Task and a Project Name** → Determine the specific task you want to accomplish with AdDownloader: only ad data retrieval, only media content retrieval, or both. Also, choose a name for your project - all downloaded content will be save in that folder.

2. 🔑 **Provide Your Access Token** → Enter your access token for authentication with the Meta Ad Library API.

3. 📋 **Set Parameters** → Answer questions or provide search parameters to tailor the data retrieval to your needs.

4. 🌐 **Call the API & Download Data** → AdDownloader calls the Meta Ad Library API and downloads the ad data, which is initially in raw JSON format.

5. 📊 **Process Data** → Convert and organize the raw data into user-friendly Excel files for easier analysis and interpretation.

6. 📥 **Download Media Content** → Access each `ad_snapshot_url` provided in the data to download the corresponding media content.

7. 📁 **Check Output** → Review all the downloaded media content and data inside the output folder with your `project_name`.

8. 📈 **Optional - Explore Insights** → Perform ad creatives (text & image) analysis and visualize the outputs in a Dashboard.

Each step in this pipeline is designed to ensure a smooth and efficient user experience, from data retrieval to processing and final output. Please note that this tool should only be used for research purposes.

## Prerequisities
AdDownloader runs on both Windows and MacOS machines. To download and run the package, several prerequisities must be fullfilled:

* Python 3.9, 3.10 or 3.11
* Verified identity and location on Facebook
* A valid Meta developer access token

## Installation

AdDownloader can be installed in three different ways: from source, from a built distribution, or as a package from *pip*.

Open a command-line tool (or a terminal) and navigate to your folder with:
```bash
cd [path-to-your-project]/AdDownloader
```
Then, create a virtual environment (venv) with:

*Windows*
```bash
python -m venv venv
```

*MacOS*
```bash
python3 -m venv myenv
```

And activate the venv:

*Windows*
```bash
venv\Scripts\activate.bat
```

*MacOS*
```bash
source myenv/bin/activate
```

### From source or built distribution
Once you're inside your repository and the virtual environment, to install from the source file run:

```bash
python -m pip install "dist/AdDownloader-0.2.7.tar.gz"
```

To install from the built distribution run:
```bash
python -m pip install "dist/AdDownloader-0.2.7-py3-none-any.whl"
```

### From pip
To install AdDownloader as a Python package run:
```bash
python -m pip install AdDownloader
```

AdDownloader is now installed and can be run!

## Usage
Once installed, AdDownloader can be run in two ways: as a command-line tool (CLI) or as a Python package.

### As a Python package
#### Download ALL ads:
1. Create an AdDownloader instance:
```bash
from AdDownloader import adlib_api
from AdDownloader.media_download import start_media_download
 
access_token = input() # the best way to upload your token
ads_api = adlib_api.AdLibAPI(access_token, project_name = "test1")
```

2. Add parameters to your search:
```bash
ads_api.add_parameters(ad_reached_countries = 'BE', ad_delivery_date_min = "2023-12-01", ad_delivery_date_max = "2023-12-31", 
                       search_terms = "McDonald's")
```
Note that `search_terms` and `search_pages_id` cannot both be empty.

3. Check the parameters and start the download of ads data:
```bash
ads_api.get_parameters()
data = ads_api.start_download()
```

4. Start the media content retrieval of the ads downloaded in the previous step:
```bash
start_media_download(project_name = "test1", nr_ads = 20, data = data)
```

#### Download POLITICAL_AND_ISSUE_ADS:
1. Create an AdDownloader instance:
```bash
from AdDownloader import adlib_api
from AdDownloader.media_download import start_media_download
 
plt_ads_api = adlib_api.AdLibAPI(access_token, project_name = "test2")
```

2. Add parameters to your search:
```bash
plt_ads_api.add_parameters(ad_reached_countries = 'US', ad_delivery_date_min = "2020-10-01", ad_delivery_date_max = "2020-10-03", 
                           ad_type = "POLITICAL_AND_ISSUE_ADS", search_page_ids = "us_parties.xlsx")
```

3. Check the parameters and start the download of ads data:
```bash
plt_ads_api.get_parameters()
plt_data = plt_ads_api.start_download()
```

4. Start the media content retrieval of the ads downloaded in the previous step:
```bash
start_media_download(project_name = "test2", nr_ads = 20, data = plt_data)
```

All the ouput can be found inside the '`output/<project_name>`' folder.

#### Run the Analytics Dashboard:
Create a dashboard with various graphs and statistics.
From the cmd/terminal: (inside your folder with venv activated)
```bash
python -m AdDownloader.app
```

From an IDE:
```bash
from AdDownloader.start_app import start_gui # takes some time to load...
start_gui()
```

Access http://127.0.0.1:8050/ once Dash is running.

### As a CLI
#### From the cmd/terminal: (inside your folder with venv activated)
```bash
python -m AdDownloader.cli
```

#### From an IDE:
```bash
from AdDownloader.cli import run_analysis
run_analysis()
```

Once the CLI tool is running, more instructions and questions will appear in the cmd/terminal that will guide the API call.

For further help and additional functionalities see the [AdDownloader documentation](https://addownloader.readthedocs.io/en/latest/index.html). 

## Image Download Setup
On some machines it might happen that a potential binary version mismatch might occur between the installed Chrome version and the required ChromeDriver. We recommend that users first try running the image download functionality of AdDownloader as it is. If an error occurs related to a version mismatch, we advise downloading the appropriate version of ChromeDriver directly from the official [ChromeDriver website](https://developer.chrome.com/docs/chromedriver/downloads) and ensuring that it matches the version of Chrome installed on their machine. Once downloaded, placing the ChromeDriver executable in a directory included in the system’s PATH should help avoid version mismatches and related errors.

## Contributing
The AdDownloader project is released with a [Contributor Code of Conduct](https://github.com/Paularossi/AdDownloader/blob/main/LICENSE.txt). By contributing to this project, you agree to abide by its terms. To contribute, follow the "forg-and-pull" Git workflow:
1. Fork the repo on GitHub and create your branch from `master`
2. Clone the project to your own machine
3. Commit changes to your own branch
4. Push your work back up to your fork
5. Submit a Pull Request so that we can review your changes

## License
This project is licensed under the GNU General Public License v3.0. You can read the license [here](https://github.com/Paularossi/AdDownloader/blob/main/LICENSE.txt). 