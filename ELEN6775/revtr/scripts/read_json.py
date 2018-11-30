#!/Users/alexstein/anaconda3/bin/python
# usage: ./scripts/read_json.py <json_file>.json | less
# function: pretty print the json file for readability
import sys, json

filename = sys.argv[1]
with open(filename, 'r') as jsonfile:
    json_data = json.load(jsonfile)
    print(json.dumps(json_data, indent=2))