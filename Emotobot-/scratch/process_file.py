__author__ = 'davidwen'

import argparse
import os
from os import listdir
from os.path import isfile, join

import emotions

def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--path', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args

def get_emotion(path):
    """
    for all the png files in path, get the emotion from the filename
    :param path: path with .png files
    :return:
    """
    onlyfiles = [join(path,f) for f in listdir(path) if isfile(join(path, f))]
    png_files = [f for f in onlyfiles if f.endswith(".png")]

    for f in png_files:
        dir, file = os.path.split(f)
        e = emotions.Emotions()
        print e.build_vector(file[0])


if __name__ == '__main__':
    args = parse_argument()
    path = args['path'][0]

    get_emotion(path)
