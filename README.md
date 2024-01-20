# AdDownloader
Add the description here

## Installation

AdDownloader can be installed in three different ways: from source, from a wheel-file, or as a package.

### Installing from source or wheel-file
Open a command-line tool (or a terminal) and navigate to your folder with

```bash
cd "[path-to-your-project]/AdDownloader"
```

Once you're inside your repository, to install from source run:

```bash
python setup.py install
```

To install from the wheel-file run:
```bash
pip install "dist/AdDownloader-0.1.0.tar.gz"
```

### Installing as a Python package
To install AdDownloader as a Python package run:
```bash
pip install "111"
```

AdDownloader is now installed and can be run!

## Usage
Once installed, AdDownloader can be run in two ways: as a command-line tool (CLI) or as a Python package.

### AdDownloader as a Python package

1. Create an AdDownloader instance:
```bash
from AdDownloader import adlib_api
from AdDownloader.media_download import start_media_download
 
ads_api = adlib_api.AdLibAPI(your-fb-access-token-here)
```

2. Add parameters to your search:
```bash
ads.add_parameters(countries = 'BE', start_date = "2023-09-01", end_date = "2023-09-02", search_terms = "pizza", project_name = "test1")
```
Note that `search_terms` and `search_pages_id` are complementary.

3. Check the parameters and start the download of ads data:
```bash
# check the parameters
ads.get_parameters()
data = ads.start_download()
```

4. Start the media content retrieval of the ads downloaded in the previous step:
```bash
start_media_download(project_name = "test1", nr_ads = 20, data = data)
```
All the ouput can be found inside the '`output/your-project-name`' folder.

### AdDownloader as a CLI
1. Open a cmd/terminal and navigate to your directory with:
```bash
cd "[path-to-your-project]/AdDownloader"
```

2. Create and activate a virtual environment (venv) with:
```bash
///
venv\Scripts\activate.bat
```

3. Finally, start the CLI tool with:
```bash
python -m AdDownloader.cli
```
Once the tool is running, more instructions and questions will appear in the cmd/terminal 

For further help see the [documentation](https://addownloader.readthedocs.io/en/latest/index.html). 

## License

This project is licensed under the GNU General Public License v3.0. 
See the LICENSE.txt file for details.