"""This module provides the AdDownloader CLI."""
"""
Created on January 11, 2024

@author: Paula G
"""

from typing import Optional
import typer
from PyInquirer import prompt, print_json, Separator
from rich import print as rprint

#from AdDownloader import __app_name__, __version__
from AdDownloader.helpers import *
from AdDownloader.media_download import *
import os
import pandas as pd
import datetime

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
        {
            'type': 'password',
            'name': 'access-token',
            'message': 'Please provide your access token',
            'when': lambda answers: answers['task'] == 'A - download ads data only'
        },
        {
            'type': 'confirm',
            'name': 'start',
            'message': 'Are you sure you want to proceed?',
        }
    ]

    answers = prompt(questions)

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

@app.command("start")
def ask_questions():
    name = typer.prompt("What is your name?")
    age = typer.prompt("How old are you?", type=int)
    favorite_color = typer.prompt("What is your favorite color?")

    print(f"Hi {name} who is {age} years old with {favorite_color} fave color!")


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