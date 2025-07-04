import sys
sys.path.append('/workspace/src/')

from database.database_setup import *
from database.model import Base, Document, Table
from dotenv import dotenv_values

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import text

import random 
import pandas as pd
import fitz 
import time
import os
import json
from io import StringIO
from tabulate import tabulate

import ipywidgets as widgets
from ipywidgets import RadioButtons, Button, VBox, Label, Layout
from IPython.display import display
from IPython.display import clear_output
from IPython.display import Image


def page_n(pdf_file, page_number, output_file="output.png", dpi=150):
    """
    Save a specific page of a PDF as a PNG.
    Args:
        pdf_file (str): Path to the PDF file.
        page_number (int): Page number to convert (0-based index).
        output_file (str): Path to save the output image.
        dpi (int): Dots per inch (resolution) for rendering the page.
    """
    # Open the PDF
    doc = fitz.open(pdf_file)
    page = doc.load_page(page_number)
    pix = page.get_pixmap(dpi=dpi)
    pix.save(output_file)
    doc.close()


def get_rating_data(tid, session, session_pdf):
    table_obj = session.query(Table).filter(Table.ir_tab_id == tid).first() 
    curr_doi = table_obj.ir_id
    page = table_obj.position_page
    try:
        t_name = table_obj.table_name
    except:
        t_name = ""
    try:
        t_caption = table_obj.caption
    except:
        t_caption = ""
    table_df = pd.DataFrame(table_obj.content, columns=list(table_obj.header))

    pdf = session_pdf.query(Document).filter(Document.doi == curr_doi).first().pdf
    with open("temp.pdf", "wb") as f:
        f.write(pdf)

    image_filename = "output.png"
    page_n("temp.pdf", page-1, output_file=image_filename, dpi=150)
    
    return table_df, t_name, t_caption, image_filename, curr_doi, tid



def log_table_feedback(doc_id, table_id, log_file='feedback_log.json'):
    """
    Displays feedback questions for a given table_id, and logs the responses to a JSON file.
    If an entry for table_id already exists in the log file, it displays a message and does not show questions.
    
    Edits:
    - If the user selects "no" for the 'recognized_table' question, the 'parse_quality' value is forced to -1.
    
    Parameters:
    - table_id (str/int): The ID of the table to log feedback for.
    - log_file (str): Path to the JSON file used for storing feedback.
    """

    # 1) Attempt to read existing data from the log file (if it exists).
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # If the file is empty or corrupted, reset data to an empty list
                data = []
    else:
        data = []

    # Ensure data is a list of feedback entries.
    if not isinstance(data, list):
        data = []
    
    # 2) Check if the table_id is already present in the log
    if any(entry.get('tid') == table_id for entry in data):
        display(Label(f"Feedback for ID '{table_id}' is already logged."))
        time.sleep(2)
        clear_output()
        return

    # 3) Create the widgets for feedback
    recognized_table_widget = RadioButtons(
        options=['yes', 'no'],
        description='Is the table presented in the dataframe a table in the PDF (and not e.g., part of references or text corpus, a figure, or a list)? ',
        value=None,
        disabled=False,
        layout=Layout(
    display='flex',
    flex_flow='column',
    align_items='stretch',
    width='100%')
    )

    parse_quality_widget = RadioButtons(
        options=['very badly', 'badly', 'ok', 'well', 'perfect'],
        description='How well was the table parsed?',
        value=None,
        disabled=False,
        layout=Layout(
    display='flex',
    flex_flow='column',
    align_items='stretch',
    width='100%')
    )

    parse_caption_widget = RadioButtons(
        options=['there is no table caption', 'caption falsely recognised (text was misinterpreted as table caption)', 'parsed badly', 'parsed ok', 'parsed perfectly'],
        description='If there is a table caption, how well was it recognized?',
        value=None,
        disabled=False,
        layout=Layout(
    display='flex',
    flex_flow='column',
    align_items='stretch',
    width='100%')
    )


    submit_button = Button(description='Submit')

    # 4) Define what happens when 'Submit' is clicked
    def on_submit_clicked(b):
        # Gather the responses
        recognized_answer = recognized_table_widget.value
        parse_quality_answer = parse_quality_widget.value
        parse_caption_answer = parse_caption_widget.value
        
        # If recognized_table is "no", log parse_quality as -1
        if recognized_answer == 'no':
            parse_quality_answer = -1
            parse_caption_answer = -1
        
        # Append the new entry
        data.append({
            'doi': doc_id,
            'tid': table_id,
            'recognized_table': recognized_answer,
            'parse_quality': parse_quality_answer,
            'caption_parsing': parse_caption_answer
        })
        
        # Write the updated data to the log file
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Display a confirmation message
        display(Label("Feedback submitted. Thank you!"))
        
        # Disable the widgets so they can't be changed after submission
        recognized_table_widget.disabled = True
        parse_quality_widget.disabled = True
        submit_button.disabled = True
        parse_caption_widget.disabled = True
        time.sleep(1)
        clear_output()
        

    # 5) Attach the click handler to the submit button
    submit_button.on_click(on_submit_clicked)

    # 6) Display the questions and the button in the notebook
    display(VBox([recognized_table_widget, parse_quality_widget, parse_caption_widget, submit_button], layout=Layout(
    display='flex',
    flex_flow='column',
    border='solid 2px',
    align_items='stretch',
    width='100%'
)))


def match_topic_table(tid, topid, session, topicdata):
    top_data = topicdata[str(topid)]
    top_data["id"] = str(topid)
    table_obj = session.query(Table).filter(Table.ir_tab_id == tid).first() 
    tab_data = {
        "doi" : table_obj.ir_id,
        "tid" : str(tid),
        "name" : table_obj.table_name,
        "caption" : table_obj.caption,
        "header" : table_obj.header, 
        "content" : table_obj.content,
        "references" : table_obj.references,
        "df" : pd.DataFrame(table_obj.content, columns=list(table_obj.header))
    }
    return top_data, tab_data

def rel_table_topic(top_data, tab_data, context=True, log_file='relass_log.json'):
    """
    Displays relevance assessment questions for given table data and a given topic data, and logs the responses to a JSON file.
    If an entry for table_id already exists in the log file, it displays a message and does not show questions.
    """

    # 1) Attempt to read existing data from the log file (if it exists).
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # If the file is empty or corrupted, reset data to an empty list
                data = []
    else:
        data = []

    # Ensure data is a list of feedback entries.
    if not isinstance(data, list):
        data = []
    
    table_id = tab_data["tid"]
    topic_id = top_data["id"]
    comp_identifier = table_id+"@"+topic_id
    # 2) Check if the table//topic combination is already present in the log
    if any(entry.get('id') == comp_identifier for entry in data):
        display(Label(f"Feedback for Table '{table_id}' and Topic '{topic_id}' is already logged."))
        time.sleep(2)
        clear_output()
        return


    # just table or also context information?
    if context:
        descr = 'How relevant are the table, its caption, and the in text refererences to the given topic?'
        tab_text = f'<b>TABLE CAPTION:<br/> {tab_data["caption"]}</b><br/><i>IN_TEXT REFERENCES:<br/> {"<br/><br/>".join(tab_data["references"])}</i><br/>'
    else:
        descr = 'How relevant is the table to the given topic?'
        tab_text = f'<br/>'


    # 3) Create the Text widgets for displaying the table and topic

    table_info_widget = widgets.HTML(
    value=tab_text,
    disabled=False 
    )  

    topic_text_widget = widgets.HTML(
    value=f'<br/><h2>TOPIC {topic_id}: {top_data["title"]}</h3><b>{top_data["description"]}</b><br/><i>{top_data["narrative"]}</i>',
    disabled=False   
    )    

    # 3) Create the widgets for feedback

    relevance_assessment_widget = RadioButtons(
        options=['not relevant', 'related, but not relevant', 'relevant', 'highly relevant'],
        description=descr,
        value=None,
        disabled=False,
        layout=Layout(
    display='flex',
    flex_flow='column',
    align_items='stretch',
    width='100%')
    )


    submit_button = Button(description='Submit')

    # 4) Define what happens when 'Submit' is clicked
    def on_submit_clicked(b):
        # Gather the responses
        relevance_assessment_answer = relevance_assessment_widget.value
        
        
        # Append the new entry
        data.append({
            'topic': top_data["id"],
            'table_id': tab_data["tid"],
            'relevance': relevance_assessment_answer
        })
        
        # Write the updated data to the log file
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Display a confirmation message
        display(Label("Label submitted. Thank you!"))
        
        # Disable the widgets so they can't be changed after submission
        relevance_assessment_widget.disabled = True
        submit_button.disabled = True
        time.sleep(1)
        clear_output()
        

    # 5) Attach the click handler to the submit button
    submit_button.on_click(on_submit_clicked)

    # 6) Display the questions and the button in the notebook
    display(VBox([table_info_widget, topic_text_widget, relevance_assessment_widget, submit_button], layout=Layout(
    display='flex',
    flex_flow='column',
    border='solid 2px',
    align_items='stretch',
    width='100%'
)))
    