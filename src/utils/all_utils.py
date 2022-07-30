import os
import logging as lg
import yaml


def read_yaml(file_path):
    with open(file_path, 'r') as f:
        content=yaml.safe_load(f)
    return content

def create_directory(dirs:list):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def log(msg,file):
    lg.basicConfig(filename=file, filemode='a', level=lg.INFO,
    format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', datefmt='%d/%m/%Y  %I:%M:%S %p')
    lg.info(msg)
