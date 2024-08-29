"""This module provides the AdDownloader Command-line Interface."""

########
# to build new distributions (new versions), in the cmd inside the venv run 'python -m build'
# to upload to PyPi, run 'python -m twine upload dist/AdDownloader-0.2.8-py3-none-any.whl'
# to install the package, inside the directory with venv run: 'python -m pip install AdDownloader'
########

import typer
import inquirer3 
from inquirer3.themes import load_theme_from_dict
from rich import print as rprint
import time
import pandas as pd

from AdDownloader.adlib_api import *
from AdDownloader.media_download import *
from AdDownloader.helpers import NumberValidator, DateValidator, CountryValidator, ExcelValidator, update_access_token

default_style = load_theme_from_dict(
    {
        "Question": {
            "mark_color": "bold_firebrick3",
            "brackets_color": "mediumpurple",
            "default_color": "bold_blue"
        },
        "List": {
            "selection_color": "bold_dodgerblue3_on_goldenrod1",
            "selection_cursor": "➤",
            "unselected_color": "dodgerblue2"
        }
    }
)


def request_params_task_AC():
    """
    Prompt user for additional parameters for the API request in tasks A and C that involve ad data download from the Meta Ad Library.

    :returns: User-provided parameters.
    :rtype: dict
    """
    add_questions = [
        inquirer3.List(
            "ad_type", 
            message="What type of ads do you want to search?", 
            choices=['All', 'Political/Elections'],
        ),
        inquirer3.Text(
            "ad_reached_countries",
            message="What reached countries do you want? (Provide the code, default is 'NL')",
            validate=CountryValidator.validate_country,
        ),
        inquirer3.Text(
            "ad_delivery_date_min",
            message="What is the minimum ad delivery date you want? (default is '2023-01-01')",
            validate=DateValidator.validate_date,
        ),
        inquirer3.Text(
            "ad_delivery_date_max",
            message="What is the maximum ad delivery date you want? (default is today\'s date)",
            validate=DateValidator.validate_date,
        ),
        inquirer3.List(
            "search_by", 
            message="Do you want to search by pages ID or by search terms?", 
            choices=['Pages ID', 'Search Terms'],
        ),
        inquirer3.Text(
            "pages_id_path",
            message="Please provide the name of your Excel file with pages ID (needs to be inside the data folder)",
            ignore=lambda answers: answers['search_by'] == 'Search Terms',
            validate=ExcelValidator.validate_excel,
        ),
        inquirer3.Text(
            "search_terms",
            message="Please provide one or more search terms, separated by a comma",
            ignore=lambda answers: answers['search_by'] == 'Pages ID',
        )
    ]

    answers = inquirer3.prompt(add_questions, theme=default_style)
    return answers


def run_task_A(project_name, answers):
    """
    Run task A: Download ads data from the Meta Online Ad Library based on user-provided parameters.

    :param project_name: The name of the current project.
    :type project_name: str
    :param answers: User's answers from the initial questions for tasks A/C regarding desired parameters.
    :type answers: dict
    """
    ads = AdLibAPI(f"{answers['access_token']}", project_name = project_name)
    # ask for search parameters
    add_answers = request_params_task_AC()
    search_by = add_answers['search_by']
    ad_type = add_answers['ad_type']
    ads.add_parameters(
        ad_reached_countries = add_answers['ad_reached_countries'], 
        ad_delivery_date_min = add_answers['ad_delivery_date_min'], 
        ad_delivery_date_max = add_answers['ad_delivery_date_max'], 
        search_page_ids = add_answers['pages_id_path'] if search_by == 'Pages ID' else None,
        search_terms = add_answers['search_terms'] if search_by == 'Search Terms' else None,
        ad_type = "ALL" if ad_type == 'All' else "POLITICAL_AND_ISSUE_ADS"
    )
        
    rprint("[yellow]Let's check the parameters you provided:[yellow]")
    rprint(f"[green bold]{ads.get_parameters()}.[green bold]")

    rprint("[yellow]Ad data download will begin now.[yellow]")
    start_time = time.time()
    ads.start_download()
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Download finished in {minutes} minutes and {seconds} seconds.")    


def run_task_B(project_name, answers):
    """
    Run task B: Download ads media content based on user's choices.

    :param project_name: The name of the current project.
    :type project_name: str
    :param answers: User's answers from the initial questions for tasks A/C regarding desired parameters.
    :type answers: dict
    :param file_name: Name of the Excel file containing ads data. If none is provided data is taken from 'output/project_name/ads/data/original_data.xlsx'
    :type file_name: str
    """
    try:
        file_path = f'output/{project_name}/ads_data/{project_name}_original_data.xlsx'
        
        data = pd.read_excel(file_path)
        total_ads = len(data)

        data = update_access_token(data, answers['access_token'])
       
        print("Starting downloading media content.")

        # if task C was chosen - continue with downloading media content
        questions_down = [
            inquirer3.List(
                "nr_ads", 
                message=f"You currently have {total_ads} ads in your search. For how many do you want to download media content?", 
                choices=['A - 50', 'B - 100', 'C - 200', 'D - insert a custom number'],
            ),
            inquirer3.Text( # if the user wants a custom number of media content
                "custom_ads_nr",
                message="Please provide the number of ads you want to download media content for",
                ignore=lambda answers: answers['nr_ads'] != 'D - insert a custom number',
                validate=NumberValidator.validate_number,
            ),
        ]
        answers_down = inquirer3.prompt(questions_down, theme=default_style)
        if answers_down["nr_ads"] == 'A - 50':
            nr_ads = 50
        elif answers_down["nr_ads"] == 'B - 100':
            nr_ads = 100
        elif answers_down["nr_ads"] == 'C - 200':
            nr_ads = 200
        else:
            nr_ads = int(answers_down["custom_ads_nr"])
        start_media_download(project_name, nr_ads=nr_ads, data=data)
    except Exception as e:
        print(e)


def intro_messages():
    """
    Display introductory messages and gather user input and Meta developer access token for selected task.
    After selecting the task, the respective function will be run.
    """
    questions = [
        inquirer3.List(
            "task", 
            message="Welcome to the AdDownloader! Select the task you want to perform: ", 
            choices=['A - download ads data only', 'B - download ads media content only', 'C - download both ads data and media content',
                     'D - open dashboard (using existing data)'],
        ),
        inquirer3.Password(
            name="access_token", 
            message='Please provide a valid access token',
            ignore=lambda answers: answers['task'] == 'D - open dashboard (using existing data)',
        ),
        inquirer3.Confirm("start", message="Are you sure you want to proceed?", default=True),
    ]

    answers = inquirer3.prompt(questions, theme=default_style)

    rprint(f"[green bold]You have chosen task {answers['task']}.[green bold]")

    if answers['task'] == 'A - download ads data only':
        rprint("[yellow]Please enter a name for your project. All created folders will use this name:[yellow]")
        project_name = input()
        run_task_A(project_name, answers)
    
    elif answers['task'] == 'B - download ads media content only':
        rprint("[yellow]Please enter the project_name you have ads data for.\n The data needs to be in the output/<project_name>/ads_data folder.[yellow]")
        project_name = input()
        run_task_B(project_name, answers)

    elif answers['task'] == 'C - download both ads data and media content':
        rprint("[yellow]Please enter a name for your project. All created folders will use this name:[yellow]")
        project_name = input()
        run_task_A(project_name, answers)
        run_task_B(project_name, answers)

    elif answers['task'] == 'D - open dashboard (using existing data)':
        rprint("[yellow]The link to open the dashboard will appear below. Click Ctrl+C to close the dashboard.[yellow]")
        from AdDownloader.start_app import start_gui # takes some time to load...
        start_gui()
        

    rprint("[yellow]=============================================[yello]")
    rprint("Finished.")

app = typer.Typer() # create the app


@app.command("run-analysis")
def run_analysis():
    """
    Main function to run the AdDownloader tool until the user stops the analysis.
    """
    
    while True:
        intro_messages()

        # ask if the user wants to perform another analysis
        rerun = typer.confirm("Do you want to perform a new analysis?")
        if not rerun:
            rprint("[yellow]=============================================[yello]")
            rprint("[yellow]Analysis completed. Thank you for using AdDownloader! [yello]")
            break


# need this to run the app
if __name__ == "__main__":
    app()   
