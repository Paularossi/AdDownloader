"""This module provides the AdDownloader CLI."""
"""
Created on January 11, 2024

@author: Paula G
"""

# to run the tool: open a cmd and go to the directory 'cd My Documents\AdDownloader'
# activate the virtual environment with 'venv\Scripts\activate.bat'
# then start the app with 'python -m AdDownloader.cli'

#from typing import Optional
import typer
from PyInquirer import prompt, print_json, Separator
from rich import print as rprint

#from AdDownloader import __app_name__, __version__
from AdDownloader.adlib_api import *
from AdDownloader.media_download import *
from AdDownloader.helpers import is_valid_excel_file
import os
import time
import pandas as pd
import logging

def request_params_task_AC():
    add_questions = [
        {
            'type': 'input',
            'name': 'countries',
            'message': 'What reached countries do you want? (Provide the code, default is "NL")'
        },
        {
            'type': 'input',
            'name': 'start_date',
            'message': 'What is the minimum ad delivery date you want? (default is "2023-01-01")'
        },
        {
            'type': 'input',
            'name': 'end_date',
            'message': 'What is the maximum ad delivery date you want? (default is today\'s date)'
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
            'when': lambda answers: answers['search_by'] == 'Pages ID'
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

    answers = prompt(add_questions)
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
    print(f"Download finished in {end_time - start_time} seconds.")
        
    rprint("[yellow]Data processing will start now.[yellow]")
    # load the data from the saved json files and process it
    final_data = transform_data(project_name, country=add_answers['countries']) 

    total_ads = len(final_data)
    print(f"Done processing and saving ads data for {total_ads} ads for project {project_name}.")


def run_task_B(project_name, answers, file_name=None):
    print("Starting downloading media content.")
    if file_name is not None:
        file_path = f'output\\{project_name}\\ads_data\\{file_name}.xlsx'
    else:
        file_path = f'output\\{project_name}\\ads_data\\original_data.xlsx'
    
    data = pd.read_excel(file_path)
    total_ads = len(data)

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
            'when': lambda answers: answers['nr_ads'] == 'D - insert a custom number'
        }
    ]
    answers_down = prompt(questions_down)
    if answers_down["nr_ads"] == 'A - 50':
        nr_ads = 50
    elif answers_down["nr_ads"] == 'B - 100':
        nr_ads = 100
    elif answers_down["nr_ads"] == 'C - 200':
        nr_ads = 200
    else:
        # check if it's a valid nr
        nr_ads = int(answers_down["custom_ads_nr"])
    start_media_download(project_name, nr_ads=nr_ads, data=data)


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
        { # if the user wants to download images
            'type': 'input',
            'name': 'data_path',
            'message': 'Please provide the path to your Excel file with ad data',
            'when': lambda answers: answers['task'] == 'B - download ads media content only'
        },
        {
            'type': 'confirm',
            'name': 'start',
            'message': 'Are you sure you want to proceed?',
        }
    ]

    answers = prompt(questions)

    rprint(f"[green bold]You have chosen task {answers['task']}.[green bold]")

    if answers['task'] == 'A - download ads data only' or answers['task'] == 'C - download both ads data and media content':
        rprint("[yellow]Please enter a name for your project. All created folders will use this name:[yellow]")
        project_name = input()
        run_task_A(project_name, answers)
    
    if answers['task'] == 'B - download ads media content only':
        rprint("[yellow]Please enter the name of the project you have ads data for.\n The data needs to be in the output\project_name\\ads_data folder.[yellow]")
        project_name = input()
        rprint("[yellow]Please enter the name of the excel file containing ads data.\n The data needs to be in the output\project_name\\ads_data folder.[yellow]")
        file_name = input()
        run_task_B(project_name, answers, file_name)

    if answers['task'] == 'C - download both ads data and media content':
        rprint("[yellow]Please enter a name for your project. All created folders will use this name:[yellow]")
        project_name = input()
        run_task_A(project_name, answers)
        run_task_B(project_name, answers)
        


    rprint("[yellow]=============================================[yello]")
    rprint("Finished.")
    #rprint("[green bold]Enter folder name :[green bold]")
    #folder_name = input()


    #subprocess.run(f"mkdir {folder_name}_created_by_{username['username']}", shell=True)

app = typer.Typer() # create the app

@app.command("run-analysis")
def run_analysis():
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

"""


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        callback=_version_callback,
        is_eager=True,
    )
) -> None:
    return

@main.command()
@click.argument('datafile')
def init(datafile):
    A command to initiate the AdDownloader project

    Paramters
    ---------
    datafile: xlsx
        Should be output data from the Facebook Ad Library
    
    """ """
    # creating project structure
    folders = ['temp', 'output']
    
    for folder in folders:
        print("Creating %s-folder" % folder)
        if os.path.exists('./%s' % folder):
            click.echo(click.style("%s-folder already exists" % folder, fg = 'yellow'))
        else:
            try:
                os.mkdir(folder)
            except OSError:
                click.echo(click.style("Creation of the directory %s failed" % folders, fg = 'red'))
            else:
                #print(Fore.GREEN + "Succesfully created %s-folder" % folder)
                click.echo(click.style("Succesfully created %s-folder" % folder, fg = "green"))
        
      
    if os.path.isfile('facebookAccessToken.txt'):
        click.echo(click.style("Warning! Credentials-file already exists", fg = "yellow"))
    else:
        print('Creating credentials-file: facebookAccessToken.txt')
        with open("facebookAccessToken.txt", 'w') as writer:
            writer.write("")
        writer.close()
        click.echo(click.style("Succesfully created credentials-file", fg = "green"))
        
        
    if os.path.isfile('metadata.txt'):
        click.echo(click.style("Warning! Metadata-file already exists", fg = "yellow"))   
    else:
        print('Creating metadata-file')
        with open("metadata.txt", 'w') as writer:
            writer.write("link, adid, content_type")
        writer.close()
        click.echo(click.style("Succesfully created metadata-file", fg = "green")) 


    # creating lists
    print('Creating internal data')
    data = pd.read_excel(str(datafile))
    adid_list = list(data['adlib_id'])
    url_list = list(data['ad_snapshot_url'])
    click.echo(click.style('Succesfully creating internal data', fg = "green")) 
    
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    with open("temp/url_list_" + timestamp + ".txt", 'w') as filehandle:
        for listitem in url_list:
            filehandle.write('%s\n' % listitem)
    
    with open("temp/adid_list_" + timestamp + ".txt", 'w') as filehandle:
        for listitem in adid_list:
            filehandle.write('%s\n' % listitem)



"""