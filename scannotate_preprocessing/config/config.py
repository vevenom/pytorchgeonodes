import ast
import configparser
from os import path, listdir

_CONVERTERS = {
    "struct": ast.literal_eval
}

def load_config(config_file):
    parser = configparser.ConfigParser(allow_no_value=True, converters=_CONVERTERS)
    parser.read([config_file])
    return parser
