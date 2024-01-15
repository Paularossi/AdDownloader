"""This module provides the AdDownloader CLI."""
"""
Created on January 11, 2024

@author: Paula G
"""

# to run the tool: open a cmd and go to the directory of the project 'cd My Documents\AdDownloader'
# activate the virtual environment with 'venv\Scripts\activate.bat'
# then start the app with 'python -m AdDownloader.cli'

#from typing import Optional
import typer
from PyInquirer import prompt, style_from_dict, Token
from prompt_toolkit.styles import Style
from rich import print as rprint

#from AdDownloader import __app_name__, __version__
from AdDownloader.adlib_api import *
from AdDownloader.media_download import *
from AdDownloader.helpers import NumberValidator, DateValidator, CountryValidator
import time
import pandas as pd
import logging

# global style for the cmd
default_style = style_from_dict({
    Token.QuestionMark: '#E91E63 bold',
    Token.Selected: '#673AB7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#2196f3 bold',
    Token.Question: '#FFD700 bold',
})


def request_params_task_AC():
    add_questions = [
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
    ads = AdLibAPI(f"{answers['access_token']}")
    # ask for search parameters
    add_answers = request_params_task_AC()
    search_by = add_answers['search_by']
    ads.add_parameters(
        countries = add_answers['countries'], 
        start_date = add_answers['start_date'], 
        end_date = add_answers['end_date'], 
        page_ids = add_answers['pages_id_path'] if search_by == 'Pages ID' else None,
        search_terms = add_answers['search_terms'] if search_by == 'Search Terms' else None,
        project_name = project_name
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


def run_task_B(project_name, file_name=None):
    try:
        if file_name is not None:
            file_path = f'output\\{project_name}\\ads_data\\{file_name}.xlsx'
        else:
            file_path = f'output\\{project_name}\\ads_data\\original_data.xlsx'
        
        data = pd.read_excel(file_path)
        total_ads = len(data)

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
    except Exception:
        pass # do nothing if no data has been downloaded


def intro_messages():
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
            ],
        },
        { # if the user wants to download ad data
            'type': 'password',
            'name': 'access_token',
            'message': 'Please provide your access token',
            'when': lambda answers: answers['task'] == 'A - download ads data only' or 
                                    answers['task'] == 'C - download both ads data and media content'
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
        rprint("[yellow]Please enter the name of the project you have ads data for.\n The data needs to be in the output\project_name\\ads_data folder.[yellow]")
        project_name = input()
        rprint("[yellow]Please enter the name of the excel file containing ads data.\n The data needs to be in the output\project_name\\ads_data folder.[yellow]")
        file_name = input()
        run_task_B(project_name, file_name)

    if answers['task'] == 'C - download both ads data and media content':
        rprint("[yellow]Please enter a name for your project. All created folders will use this name:[yellow]")
        project_name = input()
        run_task_A(project_name, answers)
        run_task_B(project_name)
        


    rprint("[yellow]=============================================[yello]")
    rprint("Finished.")

app = typer.Typer() # create the app


@app.command("run-analysis")
def run_analysis():
    #TODO: add logging tracking?
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
