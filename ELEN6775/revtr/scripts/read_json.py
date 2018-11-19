#!/Users/alexstein/anaconda3/bin/python

import sys, json

filename = sys.argv[1]
with open(filename, 'r') as jsonfile:
    json_data = json.load(jsonfile)
    print(json.dumps(json_data, indent=2))