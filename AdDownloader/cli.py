"""This module provides the AdDownloader Command-line Interface."""
"""
Created on January 11, 2024

@author: Paula G
"""
########
# to run the tool: open a cmd and go to the directory of the project 'cd My Documents\AdDownloader'
# inside your directory, create a virtual environment with 'python -m venv venv'
# activate the virtual environment with 'venv\Scripts\activate.bat'
# then start the app with 'python -m AdDownloader.cli'

# to build new distributions (new versions), in the cmd inside the venv run 'python -m build'
# then, to upload the dist archives to TestPyPi, run 'python -m twine upload --repository testpypi dist/*'
# to upload to PyPi, run 'python -m twine upload dist/AdDownloader-0.2.4-py3-none-any.whl'
# to install the package, inside the directory with venv run: 'python -m pip install AdDownloader'

# any time you change the source of your project or the configuration inside pyproject.toml, 
# you need to rebuild these files again before you can distribute the changes to PyPI.

# to generate the sphinx documentation run 'sphinx-build -M html docs/source docs'
########

import typer
from PyInquirer import prompt, style_from_dict, Token
from rich import print as rprint

from AdDownloader.adlib_api import *
from AdDownloader.media_download import *
from AdDownloader.helpers import NumberValidator, DateValidator, CountryValidator, update_access_token
import time
import pandas as pd

# global style for the cmd
default_style = style_from_dict({
    Token.QuestionMark: '#E91E63 bold',
    Token.Selected: '#673AB7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#2196f3 bold',
    Token.Question: '#FFD700 bold',
})


#TODO: add option to go back or change already defined parameters
def request_params_task_AC():
    """
    Prompt user for additional parameters for the API request in tasks A and C that involve ad data download from the Meta Ad Library.

    :returns: User-provided parameters.
    :rtype: dict
    """
    add_questions = [
        {
            'type': 'list',
            'name': 'ad_type',
            'message': 'What type of ads do you want to search?',
            'choices': ['All', 'Political/Elections']
        },
        {
            'type': 'input',
            'name': 'countries',
            'message': 'What reached countries do you want? (Provide the code, default is "NL")',
            'validate': CountryValidator
        },
        {
            'type': 'input',
            'name': 'start_date',
            'message': 'What is the minimum ad delivery date you want? (default is "2023-01-01")',
            'validate': DateValidator
        },
        {
            'type': 'input',
            'name': 'end_date',
            'message': 'What is the maximum ad delivery date you want? (default is today\'s date)',
            'validate': DateValidator
        },
        {
            'type': 'list',
            'name': 'search_by',
            'message': 'Do you want to search by pages ID or by search terms?',
            'choices': ['Pages ID', 'Search Terms']
        },
        {
            'type': 'input',
            'name': 'pages_id_path',
            'message': 'Please provide the name of your Excel file with pages ID (needs to be inside the data folder)',
            'when': lambda answers: answers['search_by'] == 'Pages ID',
        },
        {
            'type': 'input',
            'name': 'search_terms',
            'message': 'Please provide one or more search terms, separated by a comma',
            'when': lambda answers: answers['search_by'] == 'Search Terms'
        }
    ]
    #TODO: check if the excel file is valid
    """
        {
            'type': 'input',
            'name': 'pages_id_path',
            'message': 'Excel file not found or unable to open. Please provide a new path',
            'when': lambda answers: (not is_valid_excel_file(answers['pages_id_path'])) and answers['search_by'] == 'Pages ID'
        },
    """

    answers = prompt(add_questions, style=default_style)
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
        countries = add_answers['countries'], 
        start_date = add_answers['start_date'], 
        end_date = add_answers['end_date'], 
        page_ids = add_answers['pages_id_path'] if search_by == 'Pages ID' else None,
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
            {
                'type': 'list',
                'name': 'nr_ads',
                'message': f'You currently have {total_ads} ads in your search. For how many do you want to download media content?',
                'choices': [
                            {
                                'name': 'A - 50',
                            },
                            {
                                'name': 'B - 100',
                            },
                            {
                                'name': 'C - 200',
                            },
                            {
                                'name': 'D - insert a custom number',
                            },
                ]
            },
            { # if the user wants a custom number of media content
                'type': 'input',
                'name': 'custom_ads_nr',
                'message': 'Please provide the number of ads you want to download media content for',
                'when': lambda answers: answers['nr_ads'] == 'D - insert a custom number',
                'validate': NumberValidator
            }
        ]
        answers_down = prompt(questions_down, style=default_style)
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
        {
            'type': 'list',
            'name': 'task',
            'message': 'Welcome to the AdDownloader! Select the task you want to perform: ',
            'choices': [
                        {
                            'name': 'A - download ads data only',
                        },
                        {
                            'name': 'B - download ads media content only',
                        },
                        {
                            'name': 'C - download both ads data and media content',
                        },
                        {
                            'name': 'D - open dashboard (using existing data)'
                        },
            ],
        },
        {
            'type': 'password',
            'name': 'access_token',
            'message': 'Please provide a valid access token',
            'when': lambda answers: answers['task'] != 'D - open dashboard (using existing data)',
        },
        {
            'type': 'confirm',
            'name': 'start',
            'message': 'Are you sure you want to proceed?',
        }
    ]

    answers = prompt(questions, style=default_style)

    rprint(f"[green bold]You have chosen task {answers['task']}.[green bold]")

    if answers['task'] == 'A - download ads data only':
        rprint("[yellow]Please enter a name for your project. All created folders will use this name:[yellow]")
        project_name = input()
        run_task_A(project_name, answers)
    
    if answers['task'] == 'B - download ads media content only':
        rprint("[yellow]Please enter the project_name you have ads data for.\n The data needs to be in the output\<project_name>\\ads_data folder.[yellow]")
        project_name = input()
        #rprint("[yellow]Please enter the name of the excel file containing ads data (without .xlsx).\n The data needs to be in the output\<project_name>\\ads_data folder.[yellow]")
        #file_name = input()
        run_task_B(project_name, answers)

    if answers['task'] == 'C - download both ads data and media content':
        rprint("[yellow]Please enter a name for your project. All created folders will use this name:[yellow]")
        project_name = input()
        run_task_A(project_name, answers)
        run_task_B(project_name, answers)

    if answers['task'] == 'D - open dashboard (using existing data)':
        rprint("[yellow]The link to open the dashboard will appear below:[yellow]")
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

        # Prompt to ask if the user wants to perform another analysis
        rerun = typer.confirm("Do you want to perform a new analysis?")
        if not rerun:
            rprint("[yellow]=============================================[yello]")
            rprint("[yellow]Analysis completed. Thank you for using AdDownloader! [yello]")
            break


# need this to run the app
if __name__ == "__main__":
    app()   
