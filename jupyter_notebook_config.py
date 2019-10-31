# Reference: https://svds.com/jupyter-notebook-best-practices-for-data-science/
import os
from subprocess import check_call
from pathlib import Path

HTML_FOLDER_CONTAINER_PATH = "html/"

def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks to html scripts"""
    if model['type'] != 'notebook':
        return # only do this for notebooks
    folder_structure() #Check folder structure
    d, fname = os.path.split(os_path)
    name, ext = os.path.splitext(fname) 
    check_call(['jupyter', 'nbconvert', '--output-dir=' + HTML_FOLDER_CONTAINER_PATH, '--to', 'html', fname], cwd=d)

def folder_structure():
    if not os.path.isdir(HTML_FOLDER_CONTAINER_PATH):
        os.mkdir(HTML_FOLDER_CONTAINER_PATH)
    return True

c.FileContentsManager.post_save_hook = post_save